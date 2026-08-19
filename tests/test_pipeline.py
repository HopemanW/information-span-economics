from info_span_econ import extract_information_features, simulate_credit_market, interaction_benchmark

def test_text_features_bounded():
    f=extract_information_features("Revenue rose 12%. Liquidity and refinancing remain important; collateral is secured.")
    assert 0 <= f.information_span <= 1
    assert 0 <= f.information_precision <= 1

def test_known_interaction_sign():
    df=simulate_credit_market(n_deals=500, seed=7)
    b=interaction_benchmark(df)
    assert b["relationship_x_span"] < 0
