from src.risk.greeks import bs_price_delta_gamma, implied_vol_newton


def test_bs_greeks_signs_and_gamma():
    S = 100.0
    K = 100.0
    T = 30 / 365.0
    r = 0.01
    q = 0.0
    sigma = 0.25
    price_c, delta_c, gamma_c = bs_price_delta_gamma(S, K, T, r, q, sigma, "CE")
    price_p, delta_p, gamma_p = bs_price_delta_gamma(S, K, T, r, q, sigma, "PE")
    assert price_c is not None and 0 < (delta_c or 0) < 1
    assert price_p is not None and -1 < (delta_p or 0) < 0
    assert (gamma_c or 0) > 0 and (gamma_p or 0) > 0


def test_implied_vol_solver():
    S = 100.0
    K = 100.0
    T = 30 / 365.0
    r = 0.01
    q = 0.0
    true_sigma = 0.3
    price, _, _ = bs_price_delta_gamma(S, K, T, r, q, true_sigma, "CE")
    iv = implied_vol_newton(price or 0.0, S, K, T, r, q, "CE", guess=0.2)
    assert iv is not None
    assert abs(iv - true_sigma) < 1e-3
