import streamlit as st
from monte_carlo_sim import european_monte_carlo, american_monte_carlo
import plotly.express as px
import pandas as pd

st.set_page_config(layout="wide")

st.title("Monte Carlo Simulation for Option Pricing")
st.write("Choose between **European (standard MC)** and **American (LSM MC)** option pricing.")

col1, col2 = st.columns([0.3, 0.7], gap='large')

with col1:
    st.write("## Input Parameters")
    pricing_method = st.selectbox("Pricing Method", ("European", "American"))
    option_type = st.selectbox("Option Type", ("Call", "Put"))  
    stock_price = st.number_input("Stock Price", value=100)
    strike_price = st.number_input("Strike Price", value=100)
    volatility = st.number_input("Volatility", value=0.2)
    risk_free_rate = st.number_input("Risk-free Rate", value=0.05)
    time_to_maturity = st.number_input("Time to Maturity (years)", value=1.0)
    num_simulations = st.number_input("Number of Simulations", value=500)

with col2:
    st.write("## Output")
    if pricing_method == "European":
        option_price, price_paths = european_monte_carlo(option_type, stock_price, strike_price, volatility, risk_free_rate, time_to_maturity, num_simulations)
    else:
        option_price, price_paths = american_monte_carlo(option_type, stock_price, strike_price, volatility, risk_free_rate, time_to_maturity, num_simulations)

    st.metric("Estimated Option Price", f"{option_price:.2f}")

    df = pd.DataFrame(price_paths).T
    df.columns = [f"Path {i+1}" for i in range(num_simulations)]
    fig = px.line(df, labels={"index": "Time Steps", "value": "Stock Price"}, title="Simulated Price Paths")
    st.plotly_chart(fig, use_container_width=True)
