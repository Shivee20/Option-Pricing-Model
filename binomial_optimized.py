import numpy as np

def binomial_option_pricing_fast(S0, K, T, r, sigma, N, option_type="call"):
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)
    disc = np.exp(-r * dt)

    # Vectorized stock prices at maturity
    j = np.arange(N+1)
    ST = S0 * (u**(N-j)) * (d**j)

    # Payoff at maturity
    if option_type == "call":
        option_values = np.maximum(ST - K, 0)
    else:
        option_values = np.maximum(K - ST, 0)

    # Backward induction (vectorized)
    for _ in range(N):
        option_values = disc * (p * option_values[:-1] + (1-p) * option_values[1:])

    return option_values[0]
