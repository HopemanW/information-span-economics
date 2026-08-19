import argparse
import pandas as pd
from info_span_econ import simulate_credit_market, interaction_benchmark, fit_causal_forest

p=argparse.ArgumentParser()
p.add_argument("--causal-forest", action="store_true")
a=p.parse_args()

df=simulate_credit_market()
b=interaction_benchmark(df)
print("Interaction benchmark")
print(b.round(4))
print("\nKey coefficient relationship_x_span:", round(float(b["relationship_x_span"]),4))

if a.causal_forest:
    est, features=fit_causal_forest(df)
    grid=df[features].copy()
    effects=est.effect(grid.to_numpy())
    df2=df.assign(cate=effects)
    print("\nMean estimated relationship effect by information-span quartile:")
    print(df2.groupby(pd.qcut(df2.information_span,4,duplicates="drop"))["cate"].mean())
