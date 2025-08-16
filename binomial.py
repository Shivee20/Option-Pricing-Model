import streamlit as st
import numpy as np
import yfinance as yf
import math
import matplotlib.pyplot as plt
import datetime

# -------------------------------
# Utility Functions
# -------------------------------

def get_risk_free_rate():
    """Fetches the current 13-week Treasury bill yield as the risk-free rate."""
    try:
        treasury_bill_symbol = '^IRX'  # 13 week T-bill yield
        treasury_bill = yf.Ticker(treasury_bill_symbol)
        # Fetch last 5 days to ensure we get data even on non-trading days
        history = treasury_bill.history(period="5d")
        if history.empty:
            st.warning("Could not fetch risk-free rate (^IRX). Using a default of 5%.")
            return 0.05
        latest_yield = history['Close'].iloc[-1]
        risk_free_rate = latest_yield / 100
        return risk_free_rate
    except Exception as e:
        st.warning(f"Error fetching risk-free rate: {e}. Using a default of 5%.")
        return 0.05


def get_historical_log_returns(ticker_symbol):
    """Fetches 1 year of historical data and calculates log returns."""
    ticker = yf.Ticker(ticker_symbol)
    historical_data = ticker.history(period="1y")
    if historical_data.empty:
        raise ValueError("Could not fetch historical data for the ticker.")
    log_returns = np.log(historical_data['Close'] / historical_data['Close'].shift(1))
    return log_returns


def calculate_up_down_moves(vol, t_step):
    """Calculates the up and down movement factors for the binomial tree."""
    u = math.exp(vol * math.sqrt(t_step))
    d = 1 / u
    return u, d


def build_binomial_tree(current_price, u, d, steps):
    """Constructs the binomial tree for the underlying stock price."""
    tree = np.zeros([steps + 1, steps + 1])
    for i in range(steps + 1):
        for j in range(i + 1):
            tree[j, i] = current_price * (u ** (i - j)) * (d ** j)
    return tree


# -------------------------------
# Option Valuation (European + American)
# ------streamlit run binomial_options.py-------------------------

def calculate_option_values(tree, strike, risk_free_rate, steps, t_step, option_type="call", option_style="european"):
    """Calculates the option values at each node using backward induction."""
    option_tree = np.zeros_like(tree)
    u, d = tree[0, 1] / tree[0, 0], tree[1, 1] / tree[0, 0]

    # Calculate payoff at the final step (expiration)
    if option_type == "call":
        option_tree[:, steps] = np.maximum(tree[:, steps] - strike, 0)
    else: # put
        option_tree[:, steps] = np.maximum(strike - tree[:, steps], 0)

    # Risk-neutral probability (Cox-Ross-Rubinstein)
    p = (math.exp(risk_free_rate * t_step) - d) / (u - d)
    discount_rate = math.exp(-risk_free_rate * t_step)

    # Backward induction to price the option at t=0
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            # Expected value of the option in the next step
            continuation_value = discount_rate * (
                p * option_tree[j, i + 1] + (1 - p) * option_tree[j + 1, i + 1]
            )
            if option_style == "american":
                # For American options, compare with intrinsic value of early exercise
                if option_type == "call":
                    intrinsic_value = max(tree[j, i] - strike, 0)
                else: # put
                    intrinsic_value = max(strike - tree[j, i], 0)
                option_tree[j, i] = max(intrinsic_value, continuation_value)
            else: # European
                option_tree[j, i] = continuation_value
    return option_tree


# -------------------------------
# Plot Functions
# -------------------------------

def plot_binomial_tree(tree):
    """Plots the stock price binomial tree."""
    steps = tree.shape[1] -1
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(steps + 1):
        for j in range(i + 1):
            ax.scatter(i, tree[j, i], color='blue', zorder=2)
            # Add price labels to nodes
            ax.text(i, tree[j, i] * 1.02, f'${tree[j, i]:.2f}', ha='center')
            if i < steps:
                # Plot lines connecting nodes
                ax.plot([i, i + 1], [tree[j, i], tree[j, i + 1]], color='black', linestyle='-', linewidth=0.5, zorder=1)
                ax.plot([i, i + 1], [tree[j, i], tree[j + 1, i + 1]], color='black', linestyle='-', linewidth=0.5, zorder=1)
    ax.set_title('Binomial Tree for Stock Prices', fontsize=16)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig


def plot_historical_data(ticker_symbol):
    """Plots the 5-year historical closing price of the stock."""
    ticker = yf.Ticker(ticker_symbol)
    historical_data_5y = ticker.history(period="5y")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(historical_data_5y.index, historical_data_5y['Close'], label='Closing Price')
    ax.set_title(f'Historical Closing Prices - Last 5 Years of {ticker_symbol}', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Stock Price ($)', fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    return fig


# -------------------------------
# Streamlit App
# -------------------------------

def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(layout="wide")
    st.title("Binomial Option Pricing Model (European & American)")

    st.sidebar.header("Inputs")
    ticker = st.sidebar.text_input("Stock Ticker Symbol (e.g., AAPL):", "AAPL").upper()

    if ticker:
        try:
            stock = yf.Ticker(ticker)
            expirations = stock.options

            if not expirations:
                st.error(f"No option expiration dates found for {ticker}. The ticker might be incorrect or have no listed options.")
                return

            # --- User Inputs ---
            selected_expiration_str = st.sidebar.selectbox("Choose Expiration Date", expirations)
            option_type = st.sidebar.selectbox("Option Type", ["Call", "Put"]).lower()
            option_style = st.sidebar.selectbox("Option Style", ["European", "American"]).lower()
            steps = st.sidebar.slider("Number of Steps (Tree Depth):", min_value=5, max_value=500, value=50, step=5)

            # --- Data Fetching & Calculations ---
            # Calculate time to expiration in years
            today = datetime.datetime.now().date()
            expiration_date = datetime.datetime.strptime(selected_expiration_str, '%Y-%m-%d').date()
            time_to_expiration = (expiration_date - today).days / 365.25

            # Fetch relevant option chain and current stock price
            option_chain = stock.option_chain(selected_expiration_str)
            chain_data = option_chain.calls if option_type == "call" else option_chain.puts
            current_price = stock.history(period="1d")['Close'].iloc[-1]

            # Let user select strike price from the correct chain
            strike = st.sidebar.selectbox("Select Strike Price:", chain_data['strike'])

            if st.sidebar.button("Calculate Option Price"):
                with st.spinner('Calculating...'):
                    # Calculate volatility from historical log returns
                    log_returns = get_historical_log_returns(ticker)
                    volatility = np.std(log_returns) * np.sqrt(252)

                    # Get risk-free rate
                    risk_free_rate = get_risk_free_rate()

                    # Time step for the binomial tree
                    t_step = time_to_expiration / steps

                    # Build trees and calculate option price
                    u, d = calculate_up_down_moves(volatility, t_step)
                    stock_price_tree = build_binomial_tree(current_price, u, d, steps)
                    option_tree = calculate_option_values(
                        stock_price_tree, strike, risk_free_rate, steps, t_step,
                        option_type=option_type, option_style=option_style
                    )
                    option_price = option_tree[0, 0]

                    # --- Display Results ---
                    st.header("Results")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Calculated Option Price", f"${option_price:.2f}")
                    col2.metric("Current Stock Price", f"${current_price:.2f}")
                    col3.metric("Annualized Volatility", f"{volatility:.2%}")
                    col4.metric("Risk-Free Rate", f"{risk_free_rate:.2%}")

                    st.info(f"The calculated price for the *{option_style.capitalize()} {option_type.capitalize()}* option is *${option_price:.2f}*.")

                    # --- Visualizations ---
                    st.header("Visualizations")
                    # Only plot the tree if steps are manageable to avoid clutter
                    if steps <= 20:
                        st.subheader("Stock Price Binomial Tree")
                        fig_tree = plot_binomial_tree(stock_price_tree)
                        st.pyplot(fig_tree)
                    else:
                        st.warning("Binomial tree plot is disabled for more than 20 steps to ensure readability.")

                    st.subheader("Historical Stock Price (5 Years)")
                    fig_historical = plot_historical_data(ticker)
                    st.pyplot(fig_historical)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.error("Please ensure the ticker symbol is correct and that it has options data available.")

if __name__ == "__main__":
    main()