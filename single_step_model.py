import streamlit as st
import numpy as np
import yfinance as yf
import math

# Used for calculating Discount Factor 
# risk free rate = rate of return of an investment with zero risk
def get_risk_free_rate():
    treasury_bill_symbol = '^IRX' # 13 week treasury bill currently about 5%
    treasury_bill = yf.Ticker(treasury_bill_symbol)
    # last close price
    latest_yield = treasury_bill.history(period="1d")['Close'].iloc[-1]
    risk_free_rate = latest_yield/100 #convert to decimal
    return risk_free_rate
r = get_risk_free_rate()

def get_data(ticker_symbol):
    #define ticker
    ticker = yf.Ticker(ticker_symbol)
    #historical data
    historical_data = ticker.history(period="1y")
    #Returns closing price
    return historical_data["Close"]

def fetch_option_data(ticker_symbol):
    # Define ticker
    ticker = yf.Ticker(ticker_symbol)
    # current price
    current_price = ticker.history(period="1d")['Close'].iloc[-1]
    #Options data (calls, puts: using options_data.calls, options_data.puts)
    options_data = ticker.option_chain()
    # Strike prices
    strikes = options_data.calls['strike']
    strike = strikes.iloc[(strikes - current_price).abs().argmin()]
    return current_price, options_data, strike

#print(fetch_option_data("AMZN"))
#print(get_data("AMZN"))
ticker = "AMZN"
historical_data = get_data(ticker)
log_returns = np.log(historical_data / historical_data.shift(1))
#print(log_returns)
#Volitility over time frame 1yr
vol = np.std(log_returns) * np.sqrt(252)  # Annualize the volatility

t_step = 1/12

def up_move(vol, t_step):
    u = (math.e)**(vol*math.sqrt(t_step))
    return u

def down_move(vol, t_step):
    d = (math.e)**(-vol*math.sqrt(t_step))
    return d

current_price, options_data, strike = fetch_option_data(ticker)

# Calculated step nodes(price movement up or down) based on volitiliy and time step(1 month) 
next_up = current_price*up_move(vol, t_step)
next_down = current_price*down_move(vol, t_step)


# CALL/PUT PAYOFF SINGLE STEP
call_payoff = max(next_up - strike, 0)
put_payoff = max(strike - next_down, 0)
# Discounted CALL/PUT Payoff using risk free rate
discounted_call = call_payoff / (1 + r * (1/12))
discounted_put = put_payoff / (1 + r * (1/12))

# delta = (call_payoff - put_payoff)/(rise_price - fall_price)
delta = (discounted_call - discounted_put)/(next_up-next_down)
# Need to buy delta% of a share to make portfolio riskless


# Call/Rise - Portfolio Value if stock rises
portfolio1 = -discounted_call + next_up*delta
# Put/Fall - Portfolio value if stock falls
portfolio2 = -discounted_put + next_down*delta

# Price to charge client for option ( V )
V = delta*current_price - portfolio2