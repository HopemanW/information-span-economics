from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier

def interaction_benchmark(df: pd.DataFrame, outcome: str = "selected") -> pd.Series:
    """OLS benchmark with relationship x span interaction."""
    y=df[outcome].to_numpy(float)
    X=np.column_stack([
        np.ones(len(df)),
        df["relationship"], df["information_span"],
        df["relationship"]*df["information_span"],
        df["information_precision"], df["risk"], df["opacity"], df["competition"],
    ])
    beta=np.linalg.lstsq(X,y,rcond=None)[0]
    names=["intercept","relationship","information_span","relationship_x_span",
           "information_precision","risk","opacity","competition"]
    return pd.Series(beta,index=names)

def fit_causal_forest(df: pd.DataFrame, outcome: str = "selected"):
    """Estimate heterogeneous relationship effects; identification must come from research design."""
    try:
        from econml.dml import CausalForestDML
    except ImportError as exc:
        raise ImportError("Install optional dependency: pip install -e '.[causal]'") from exc
    features=["information_span","information_precision","risk","opacity","competition"]
    X=df[features].to_numpy(float)
    T=df["relationship"].to_numpy(float)
    Y=df[outcome].to_numpy(float)
    est=CausalForestDML(
        model_y=RandomForestRegressor(n_estimators=200,min_samples_leaf=10,random_state=1),
        model_t=RandomForestClassifier(n_estimators=200,min_samples_leaf=10,random_state=2),
        n_estimators=400, min_samples_leaf=20, random_state=3,
    )
    est.fit(Y,T,X=X)
    return est, features
