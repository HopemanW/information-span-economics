from __future__ import annotations
import numpy as np
import pandas as pd

def simulate_credit_market(n_deals: int = 1200, n_banks: int = 6, seed: int = 20260819) -> pd.DataFrame:
    """Synthetic bank-deal panel with a known information-span mechanism."""
    rng=np.random.default_rng(seed)
    rows=[]
    bank_specialization=rng.normal(0, 0.5, n_banks)
    for d in range(n_deals):
        risk=rng.normal()
        opacity=np.clip(rng.beta(2,2), 0, 1)
        competition=np.clip(rng.beta(2,2), 0, 1)
        span=np.clip(0.20 + 0.55*(1-opacity) + 0.20*competition + rng.normal(0,0.12),0,1)
        precision=np.clip(0.35 + 0.35*(1-opacity) + rng.normal(0,0.12),0,1)
        for b in range(n_banks):
            rel=rng.binomial(1, 0.18 + 0.22*(1-opacity))
            # Relationship advantage is deliberately weakened by broader information span.
            latent=(
                -0.45 + 0.55*rel - 0.85*rel*span + 0.25*precision
                - 0.35*risk + 0.25*competition + bank_specialization[b] + rng.normal(0,0.55)
            )
            selected=int(latent>0)
            spread=(
                220 + 55*risk + 25*opacity - 18*competition - 20*precision
                - 28*rel + 45*rel*span + rng.normal(0,18)
            )
            rows.append({"deal_id":d,"bank_id":b,"risk":risk,"opacity":opacity,
                         "competition":competition,"information_span":span,
                         "information_precision":precision,"relationship":rel,
                         "selected":selected,"spread_bps":spread})
    return pd.DataFrame(rows)
