from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Mapping, Optional
import re
import numpy as np
import pandas as pd

DIMENSIONS: Mapping[str, tuple[str, ...]] = {
    "operating_performance": ("revenue", "sales", "margin", "ebitda", "profit", "cash flow"),
    "collateral_quality": ("collateral", "secured", "asset coverage", "property", "inventory"),
    "liquidity_refinancing": ("liquidity", "refinancing", "maturity", "revolver", "credit line"),
    "management_organization": ("management", "governance", "organization", "strategy", "execution"),
    "covenant_risk": ("covenant", "leverage ratio", "coverage ratio", "default", "waiver"),
    "industry_demand": ("industry", "market demand", "orders", "backlog", "competition", "market share"),
}

@dataclass(frozen=True)
class InformationFeatures:
    information_span: float
    information_precision: float
    dimensions_covered: int
    n_dimensions: int
    quantitative_density: float

def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"[.!?;\n]+", text) if s.strip()]

def _offline_scores(text: str) -> dict[str, float]:
    lower = text.lower()
    scores = {}
    for name, words in DIMENSIONS.items():
        hits = sum(1 for w in words if w in lower)
        scores[name] = float(min(1.0, hits / 2.0))
    return scores

def _quantitative_density(text: str) -> float:
    sents = _sentences(text)
    if not sents:
        return 0.0
    return sum(bool(re.search(r"\b\d+(?:\.\d+)?%?\b", s)) for s in sents) / len(sents)

def extract_information_features(
    text: str,
    scorer: Optional[Callable[[str], Mapping[str, float]]] = None,
    threshold: float = 0.5,
) -> InformationFeatures:
    text = "" if text is None else str(text)
    raw = _offline_scores(text) if scorer is None else dict(scorer(text))
    scores = {k: float(np.clip(raw.get(k, 0.0), 0.0, 1.0)) for k in DIMENSIONS}
    covered = [v for v in scores.values() if v >= threshold]
    q = _quantitative_density(text)
    span = len(covered) / len(DIMENSIONS)
    precision = float(np.mean(covered)) if covered else 0.0
    precision = float(np.clip(0.85 * precision + 0.15 * q, 0.0, 1.0)) if covered else 0.0
    return InformationFeatures(span, precision, len(covered), len(DIMENSIONS), q)

def transform_text_frame(df: pd.DataFrame, *, id_col: str, text_col: str, scorer=None) -> pd.DataFrame:
    if id_col not in df or text_col not in df:
        raise KeyError("id_col and text_col must exist")
    rows=[]
    for i, txt in zip(df[id_col], df[text_col]):
        f=extract_information_features(txt, scorer=scorer)
        rows.append({id_col:i, "information_span":f.information_span,
                     "information_precision":f.information_precision,
                     "dimensions_covered":f.dimensions_covered,
                     "quantitative_density":f.quantitative_density})
    return pd.DataFrame(rows)
