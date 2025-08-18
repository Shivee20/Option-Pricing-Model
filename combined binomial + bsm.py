import streamlit as st
import numpy as np
import pandas as pd
import yfinance as yf
import math
import matplotlib.pyplot as plt
from datetime import datetime, date
from scipy.stats import norm

# =========================
# Constants
# =========================
DAYS_PER_YEAR = 365.25

# =========================
# Utilities: Data & Rates
# =========================
@st.cache_data(show_spinner=False, ttl=60*30)
def get_risk_free_rate():
    """Fetch current 13-week T-bill yield (^IRX) as annual risk-free rate (decimal)."""
    try:
        tbill = yf.Ticker("^IRX").history(period="5d")
        if tbill.empty:
            return 0.05
        return float(tbill["Close"].iloc[-1]) / 100.0
    except Exception:
        return 0.05

@st.cache_data(show_spinner=False, ttl=60*30)
def get_historical_log_returns(ticker):
    """1y daily log returns for volatility estimation."""
    hist = yf.Ticker(ticker).history(period="1y")
    if hist.empty or "Close" not in hist.columns:
        raise ValueError("No historical data available for ticker.")
    lr = np.log(hist["Close"] / hist["Close"].shift(1)).dropna()
    return lr

@st.cache_data(show_spinner=False, ttl=60*30)
def get_5y_prices(ticker):
    hist = yf.Ticker(ticker).history(period="5y")
    return hist

# =========================
# Black–Scholes Model (European)
# =========================
class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.validate(S, K, T, r, sigma)
        self.S, self.K, self.T, self.r, self.sigma = S, K, T, r, sigma

    @staticmethod
    def validate(S, K, T, r, sigma):
        if S <= 0 or K <= 0: raise ValueError("S and K must be positive.")
        if T <= 0: raise ValueError("T must be positive.")
        if r < 0 or r > 1: raise ValueError("r must be between 0 and 1.")
        if sigma <= 0 or sigma > 3: raise ValueError("sigma must be >0 and reasonable.")

    def d1(self):
        return (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))

    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_price(self):
        return self.S * norm.cdf(self.d1()) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2())

    def put_price(self):
        return self.K * np.exp(-self.r * self.T) - self.S + self.call_price()

    # Greeks (vega/theta/rho scaled "per 1%" change)
    def call_delta(self): return norm.cdf(self.d1())
    def put_delta(self):  return -norm.cdf(-self.d1())
    def gamma(self):      return norm.pdf(self.d1()) / (self.S * self.sigma * np.sqrt(self.T))
    def vega(self):       return 0.01 * self.S * norm.pdf(self.d1()) * np.sqrt(self.T)
    def call_theta(self): return 0.01 * (-(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T)) - self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2()))
    def put_theta(self):  return 0.01 * (-(self.S * norm.pdf(self.d1()) * self.sigma) / (2 * np.sqrt(self.T)) + self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2()))
    def call_rho(self):   return 0.01 * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(self.d2())
    def put_rho(self):    return -0.01 * self.K * self.T * np.exp(-self.r * self.T) * norm.cdf(-self.d2())

# =========================
# Binomial Model (European & American)
# =========================
def calculate_up_down_moves(vol, dt):
    u = math.exp(vol * math.sqrt(dt))
    d = 1 / u
    return u, d

def build_binomial_tree(S0, u, d, steps):
    tree = np.zeros((steps+1, steps+1))
    for i in range(steps+1):
        for j in range(i+1):
            tree[j, i] = S0 * (u ** (i - j)) * (d ** j)
    return tree

def price_binomial(tree, K, r, steps, dt, option_type="call", option_style="european"):
    """Backward induction with early-exercise check for American."""
    opt = np.zeros_like(tree)
    u, d = tree[0, 1] / tree[0, 0], tree[1, 1] / tree[0, 0]
    p = (math.exp(r * dt) - d) / (u - d)
    disc = math.exp(-r * dt)

    # Payoff at maturity
    if option_type == "call":
        opt[:, steps] = np.maximum(tree[:, steps] - K, 0.0)
    else:
        opt[:, steps] = np.maximum(K - tree[:, steps], 0.0)

    # Roll back
    for i in range(steps-1, -1, -1):
        for j in range(i+1):
            cont = disc * (p * opt[j, i+1] + (1 - p) * opt[j+1, i+1])
            if option_style == "american":
                intrinsic = (tree[j, i] - K) if option_type == "call" else (K - tree[j, i])
                intrinsic = max(intrinsic, 0.0)
                opt[j, i] = max(intrinsic, cont)
            else:
                opt[j, i] = cont
    return opt

# =========================
# Plot helpers
# =========================
def plot_binomial_tree(tree):
    steps = tree.shape[1] - 1
    fig, ax = plt.subplots(figsize=(11, 7))
    for i in range(steps+1):
        for j in range(i+1):
            ax.scatter(i, tree[j, i], zorder=2)
            ax.text(i, tree[j, i] * 1.01, f'{tree[j, i]:.2f}', ha='center', fontsize=8)
            if i < steps:
                ax.plot([i, i+1], [tree[j, i], tree[j, i+1]], linewidth=0.6, zorder=1)
                ax.plot([i, i+1], [tree[j, i], tree[j+1, i+1]], linewidth=0.6, zorder=1)
    ax.set_title('Binomial Tree (Underlying Price)')
    ax.set_xlabel('Step')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

def plot_5y_history(hist, ticker):
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(hist.index, hist["Close"])
    ax.set_title(f'{ticker} — 5Y Closing Price')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price')
    ax.grid(True, linestyle='--', alpha=0.5)
    return fig

# =========================
# Streamlit App
# =========================
def main():
    st.set_page_config(layout="wide", page_title="Options Dashboard (Binomial + Black–Scholes)")
    st.title("Options Pricing Dashboard — Binomial (Eur/Amer) + Black–Scholes (Eur) + Greeks")

    # ---------- Sidebar Inputs ----------
    st.sidebar.header("Inputs")

    data_mode = st.sidebar.radio("Data Source", ["Use Ticker (yfinance)", "Manual Inputs"], index=0)

    if data_mode == "Use Ticker (yfinance)":
        ticker = st.sidebar.text_input("Ticker (e.g., AAPL)", "AAPL").upper()

        # Load options metadata
        expirations = []
        try:
            expirations = yf.Ticker(ticker).options
        except Exception:
            pass

        if not expirations:
            st.sidebar.error("No option expirations found. Switch to Manual Inputs or try another ticker.")
            use_chain = False
        else:
            use_chain = True

        if use_chain:
            exp_str = st.sidebar.selectbox("Expiration", expirations)
            chain = yf.Ticker(ticker).option_chain(exp_str)
            opt_type = st.sidebar.selectbox("Option Type", ["Call", "Put"]).lower()
            strikes = (chain.calls["strike"] if opt_type == "call" else chain.puts["strike"]).values
            K = float(st.sidebar.selectbox("Strike", strikes))
            # Spot & T
            spot = float(yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1])
            T_days = (datetime.strptime(exp_str, "%Y-%m-%d").date() - date.today()).days
            T = max(T_days, 1) / DAYS_PER_YEAR
        else:
            # fallback manual
            spot = st.sidebar.number_input("Spot", 100.0, step=0.1)
            K = st.sidebar.number_input("Strike", 100.0, step=0.1)
            T = st.sidebar.number_input("Time to Maturity (years)", 0.5, min_value=0.001, step=0.01)
            opt_type = st.sidebar.selectbox("Option Type", ["Call", "Put"]).lower()

    else:
        ticker = None
        spot = st.sidebar.number_input("Spot", 100.0, min_value=0.0001, step=0.1)
        K = st.sidebar.number_input("Strike", 100.0, min_value=0.0001, step=0.1)
        T = st.sidebar.number_input("Time to Maturity (years)", 0.25, min_value=0.001, step=0.01)
        opt_type = st.sidebar.selectbox("Option Type", ["Call", "Put"]).lower()

    opt_style = st.sidebar.selectbox("Option Style", ["European", "American"]).lower()
    steps = st.sidebar.slider("Binomial Steps", 5, 500, 50, step=5)

    # Volatility source
    vol_mode = st.sidebar.radio("Volatility", ["Historical (1y)", "Manual %"], index=0)
    if vol_mode == "Historical (1y)" and ticker:
        try:
            lr = get_historical_log_returns(ticker)
            sigma = float(np.std(lr) * np.sqrt(252))
            st.sidebar.write(f"Hist. σ ≈ {sigma:.2%}")
        except Exception as e:
            st.sidebar.warning(f"Could not compute historical volatility: {e}")
            sigma = st.sidebar.number_input("Volatility (%)", 20.0, min_value=0.01, step=0.1) / 100.0
    else:
        sigma = st.sidebar.number_input("Volatility (%)", 20.0, min_value=0.01, step=0.1) / 100.0

    # Risk-free rate source
    rf_mode = st.sidebar.radio("Risk-free Rate", ["From ^IRX", "Manual %"], index=0)
    if rf_mode == "From ^IRX":
        r = get_risk_free_rate()
        st.sidebar.write(f"r ≈ {r:.2%}")
    else:
        r = st.sidebar.number_input("Risk-free Rate (%)", 5.0, min_value=0.0, step=0.1) / 100.0

    calc = st.sidebar.button("Calculate")

    # ---------- Main Output ----------
    if not calc:
        st.info("Set your inputs in the sidebar and click **Calculate**.")
        return

    # Binomial build & price
    dt = T / steps
    u, d = calculate_up_down_moves(sigma, dt)
    tree = build_binomial_tree(spot, u, d, steps)
    opt_tree = price_binomial(tree, K, r, steps, dt, option_type=opt_type, option_style=opt_style)
    binom_price = float(opt_tree[0, 0])

    # Black–Scholes (only if European)
    bsm_price = None
    greeks_table = None
    if opt_style == "european":
        bsm = BlackScholesModel(spot, K, T, r, sigma)
        if opt_type == "call":
            bsm_price = bsm.call_price()
            greeks_table = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Vega (per 1%)", "Theta (per 1%)", "Rho (per 1%)"],
                "Value": [bsm.call_delta(), bsm.gamma(), bsm.vega(), bsm.call_theta(), bsm.call_rho()]
            })
        else:
            bsm_price = bsm.put_price()
            greeks_table = pd.DataFrame({
                "Greek": ["Delta", "Gamma", "Vega (per 1%)", "Theta (per 1%)", "Rho (per 1%)"],
                "Value": [bsm.put_delta(), bsm.gamma(), bsm.vega(), bsm.put_theta(), bsm.put_rho()]
            })

    # ---- KPIs ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Binomial Price", f"${binom_price:,.2f}")
    c2.metric("Spot (S)", f"${spot:,.2f}")
    c3.metric("Volatility (σ)", f"{sigma:.2%}")
    c4.metric("Risk‑free (r)", f"{r:.2%}")

    if bsm_price is not None:
        c5, c6 = st.columns(2)
        c5.metric("Black–Scholes Price", f"${bsm_price:,.2f}")
        c6.metric("Binomial − BSM", f"${(binom_price - bsm_price):,.2f}")

    # ---- Greeks ----
    if greeks_table is not None:
        st.subheader("Greeks (European, Black–Scholes)")
        st.dataframe(greeks_table, use_container_width=True)

    # ---- Visualizations ----
    st.subheader("Visualizations")

    # (A) Binomial Tree (only for small steps)
    if steps <= 20:
        st.caption("Binomial Tree (small step counts only for clarity)")
        fig_tree = plot_binomial_tree(tree)
        st.pyplot(fig_tree)
    else:
        st.caption("Binomial tree hidden for steps > 20 (too dense).")

    # (B) Price vs Stock (use BSM if European, else compute via Binomial sweep)
    st.markdown("**Option Price vs Stock (S)**")
    S_grid = np.linspace(0.5 * spot, 1.5 * spot, 60)
    prices_vs_S = []
    if opt_style == "european":
        for Sg in S_grid:
            m = BlackScholesModel(Sg, K, T, r, sigma)
            prices_vs_S.append(m.call_price() if opt_type == "call" else m.put_price())
    else:
        # American: recompute binomial along the S-grid (coarser but informative)
        for Sg in S_grid:
            t = build_binomial_tree(Sg, u, d, steps)
            opt_t = price_binomial(t, K, r, steps, dt, option_type=opt_type, option_style="american")
            prices_vs_S.append(opt_t[0, 0])

    figS, axS = plt.subplots(figsize=(10, 4))
    axS.plot(S_grid, prices_vs_S)
    axS.set_xlabel("Stock Price (S)")
    axS.set_ylabel("Option Price")
    axS.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(figS)

    # (C) Sensitivity vs Volatility (σ)
    st.markdown("**Option Price vs Volatility (σ)**")
    vol_grid = np.linspace(max(0.01, sigma * 0.5), min(3.0, sigma * 1.5), 40)
    prices_vs_vol = []
    if opt_style == "european":
        for vg in vol_grid:
            m = BlackScholesModel(spot, K, T, r, vg)
            prices_vs_vol.append(m.call_price() if opt_type == "call" else m.put_price())
    else:
        for vg in vol_grid:
            u2, d2 = calculate_up_down_moves(vg, dt)
            t2 = build_binomial_tree(spot, u2, d2, steps)
            opt_t2 = price_binomial(t2, K, r, steps, dt, option_type=opt_type, option_style="american")
            prices_vs_vol.append(opt_t2[0, 0])

    figV, axV = plt.subplots(figsize=(10, 4))
    axV.plot(vol_grid, prices_vs_vol)
    axV.set_xlabel("Volatility (σ)")
    axV.set_ylabel("Option Price")
    axV.grid(True, linestyle='--', alpha=0.5)
    st.pyplot(figV)

    # (D) Historical 5y (if ticker mode)
    if data_mode == "Use Ticker (yfinance)" and ticker:
        try:
            hist5 = get_5y_prices(ticker)
            if not hist5.empty:
                st.markdown("**5‑Year Historical Price**")
                figH = plot_5y_history(hist5, ticker)
                st.pyplot(figH)
        except Exception as e:
            st.warning(f"Could not plot 5y history: {e}")

if __name__ == "__main__":
    main()
