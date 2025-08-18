import numpy as np
import math

# Function to calculate payoff
def calculate_option_payoff(option_type, stock_price, strike_price):
    if option_type == "Call":
        return np.maximum(stock_price - strike_price, 0)
    elif option_type == "Put":
        return np.maximum(strike_price - stock_price, 0)

# ---------------- European Option (Standard Monte Carlo) ---------------- #
def european_monte_carlo(option_type, stock_price, strike_price, volatility, risk_free_rate, time_to_maturity, num_simulations):
    dt = time_to_maturity / 252  # daily steps
    option_payoffs = []
    price_paths = []

    for _ in range(num_simulations):
        price_path = []
        stock_price_copy = stock_price

        for _ in range(int(252 * time_to_maturity)):
            drift = (risk_free_rate - 0.5 * volatility**2) * dt
            shock = volatility * math.sqrt(dt) * np.random.normal(0, 1)
            stock_price_copy *= math.exp(drift + shock)
            price_path.append(stock_price_copy)

        price_paths.append(price_path)
        option_payoff = calculate_option_payoff(option_type, stock_price_copy, strike_price)
        option_payoffs.append(option_payoff)

    option_price = np.exp(-risk_free_rate * time_to_maturity) * np.mean(option_payoffs)
    return option_price, price_paths

# ---------------- American Option (LSM Monte Carlo) ---------------- #
def american_monte_carlo(option_type, stock_price, strike_price, volatility, risk_free_rate, time_to_maturity, num_simulations):
    steps = 50
    dt = time_to_maturity / steps

    price_paths = np.zeros((num_simulations, steps + 1))
    price_paths[:, 0] = stock_price

    for t in range(1, steps + 1):
        z = np.random.normal(0, 1, num_simulations)
        price_paths[:, t] = price_paths[:, t - 1] * np.exp(
            (risk_free_rate - 0.5 * volatility**2) * dt + volatility * np.sqrt(dt) * z
        )

    payoffs = calculate_option_payoff(option_type, price_paths[:, -1], strike_price)

    for t in range(steps - 1, 0, -1):
        stock_prices_t = price_paths[:, t]
        exercise_values = calculate_option_payoff(option_type, stock_prices_t, strike_price)
        itm = exercise_values > 0

        if np.any(itm):
            discounted_future_values = payoffs * np.exp(-risk_free_rate * dt)
            coeffs = np.polyfit(stock_prices_t[itm], discounted_future_values[itm], 2)
            continuation_values = np.polyval(coeffs, stock_prices_t)

            exercise = exercise_values > continuation_values
            payoffs[itm & exercise] = exercise_values[itm & exercise]
            payoffs[itm & ~exercise] = discounted_future_values[itm & ~exercise]
        else:
            payoffs = payoffs * np.exp(-risk_free_rate * dt)

    option_price = np.mean(payoffs) * np.exp(-risk_free_rate * dt)
    return option_price, price_paths.tolist()
