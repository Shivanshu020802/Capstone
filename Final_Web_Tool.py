# Final_Web_Tool.py

import os
import sys
import json
import re
from openai import OpenAI
import time
import pandas as pd
import numpy as np
import math
import warnings
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf
import streamlit as st

# ==============================================================================
# PART 1: GENAI CONVERSATIONAL INTERFACE (ADAPTED FOR STREAMLIT)
# ==============================================================================

def get_chatbot_response(user_input, conversation_history):
    """
    Sends user input and conversation history to GenAI to get a response.
    """
    try:
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if not client.api_key:
            st.error("Error: OPENAI_API_KEY environment variable not set.")
            return "Please set your OpenAI API key as an environment variable.", None
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return "An error occurred. Please check your API key and connection.", None

    conversation_history.append({"role": "user", "content": user_input})

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=conversation_history,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        st.error(f"Error getting AI response: {e}")
        return "An error occurred with the AI. Please try again.", None

    parsing_prompt = conversation_history + [
        {"role": "system", "content": """
            Based on the entire conversation so far, extract the four required values and return them in a JSON object. If a value is still missing, use null.
            Do not include any other text.
            Example JSON: {"capital_amount": 2000000, "time_horizon_years": 3, "risk_tolerance_loss_pct": 10, "investment_style": "Moderate", "FD_rate": 6.5}
            """}
    ]
    try:
        parsing_response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=parsing_prompt,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        parsed_data = json.loads(parsing_response.choices[0].message.content)
        return ai_response, parsed_data
    except Exception as e:
        return ai_response, None

# ==============================================================================
# PART 2: CORE LOGIC (ADAPTED FROM 'THE FINAL DAY.py')
# ==============================================================================

EXCEL_INPUT = "Nifty50_Portfolio_Analytics.xlsx"
EXCEL_OUTPUT = "Portfolio_Optimization_Output.xlsx"
SHEET_BETAS = "Betas"
SHEET_COV = "Covariance_Matrix"
SHEET_RET = "Monthly_Returns"
RFR_TICKER = "^IRX"
DROP_ASSETS_CONTAINING = ["NIFTY_50", "TBILL"]
PRICE_LOOKBACK_DAYS = 5
VAR_ALPHA = 0.95

# The following functions are from your 'THE FINAL DAY.py' and are now correctly integrated.
def map_risk_aversion_from_var(MAX_LOSS_TOLERANCE):
    if MAX_LOSS_TOLERANCE >= 0.50:
        return 1
    elif MAX_LOSS_TOLERANCE >= 0.40:
        return 2
    elif MAX_LOSS_TOLERANCE >= 0.35:
        return 3
    elif MAX_LOSS_TOLERANCE >= 0.30:
        return 4
    elif MAX_LOSS_TOLERANCE >= 0.25:
        return 5
    elif MAX_LOSS_TOLERANCE >= 0.20:
        return 6
    elif MAX_LOSS_TOLERANCE >= 0.15:
        return 7
    elif MAX_LOSS_TOLERANCE >= 0.10:
        return 8
    elif MAX_LOSS_TOLERANCE >= 0.05:
        return 9
    else:
        return 10

def generate_portfolio_analytics(time_horizon_months):
    nifty50_tickers = [
        "ADANIENT.NS", "ADANIPORTS.NS", "APOLLOHOSP.NS", "ASIANPAINT.NS", "AXISBANK.NS",
        "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BEL.NS", "BHARTIARTL.NS",
        "CIPLA.NS", "COALINDIA.NS", "DRREDDY.NS", "EICHERMOT.NS", "GRASIM.NS",
        "HCLTECH.NS", "HDFCBANK.NS", "HDFCLIFE.NS", "HEROMOTOCO.NS", "HINDALCO.NS",
        "HINDUNILVR.NS", "ICICIBANK.NS", "INDUSINDBK.NS", "INFY.NS", "ITC.NS",
        "JIOFIN.NS", "JSWSTEEL.NS", "KOTAKBANK.NS", "LT.NS", "M&M.NS",
        "MARUTI.NS", "NESTLEIND.NS", "NTPC.NS", "ONGC.NS", "POWERGRID.NS",
        "RELIANCE.NS", "SBILIFE.NS", "SBIN.NS", "SHRIRAMFIN.NS", "SUNPHARMA.NS",
        "TCS.NS", "TATACONSUM.NS", "TATAMOTORS.NS", "TATASTEEL.NS", "TECHM.NS",
        "TITAN.NS", "TRENT.NS", "ULTRACEMCO.NS", "WIPRO.NS", "NIFTY_50"
    ]
    all_tickers = [yf.Ticker(t) for t in nifty50_tickers]
    data = yf.download([t.ticker for t in all_tickers], period='max')['Close']
    data = data.rename(columns={t.ticker: t.ticker for t in all_tickers})
    data.dropna(inplace=True)
    monthly_prices = data.resample('M').last()
    monthly_returns = monthly_prices.pct_change().dropna()
    returns_subset = monthly_returns.tail(time_horizon_months)
    betas = {}
    nifty_returns = returns_subset['NIFTY_50']
    for col in returns_subset.columns:
        if col == 'NIFTY_50':
            continue
        cov_val = np.cov(returns_subset[col].dropna(), nifty_returns.dropna())[0][1]
        var_val = np.var(nifty_returns.dropna())
        beta = cov_val / var_val if var_val != 0 else np.nan
        betas[col] = beta
    beta_df = pd.DataFrame.from_dict(betas, orient='index', columns=['Beta'])
    cov_matrix = returns_subset.cov()
    with pd.ExcelWriter(EXCEL_INPUT, engine='openpyxl') as writer:
        monthly_returns.to_excel(writer, sheet_name='Monthly_Returns')
        returns_subset.to_excel(writer, sheet_name=f'Returns_Last_{time_horizon_months}M')
        beta_df.to_excel(writer, sheet_name='Betas')
        cov_matrix.to_excel(writer, sheet_name='Covariance_Matrix')

def read_inputs():
    if not os.path.exists(EXCEL_INPUT):
        raise FileNotFoundError(f"Could not find '{EXCEL_INPUT}' in current folder.")
    xl = pd.ExcelFile(EXCEL_INPUT)
    betas = pd.read_excel(xl, SHEET_BETAS, index_col=0)
    cov = pd.read_excel(xl, SHEET_COV, index_col=0)
    monthly_returns = pd.read_excel(xl, SHEET_RET, index_col=0, parse_dates=True)
    cov = cov.loc[cov.index.intersection(cov.columns), cov.columns.intersection(cov.index)]
    assets = list(cov.columns)
    investable = []
    for a in assets:
        if any(tag in a.upper() for tag in [s.upper() for s in DROP_ASSETS_CONTAINING]):
            continue
        investable.append(a)
    beta_series = betas.iloc[:, 0] if betas.shape[1] == 1 else betas.squeeze()
    beta_series = beta_series.reindex(investable).astype(float)
    cov_invest = cov.loc[investable, investable]
    return beta_series, cov_invest, monthly_returns

def fetch_latest_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            hist = yf.download(t, period=f"{PRICE_LOOKBACK_DAYS}d", interval="1d", progress=False)["Close"].dropna()
            if not hist.empty:
                prices[t] = float(hist.iloc[-1])
            else:
                prices[t] = np.nan
        except Exception:
            prices[t] = np.nan
    return pd.Series(prices, name="LastPrice")

def solve_continuous_utility(mu, cov, A, max_loss_tolerance, var_alpha, equity_indices, max_equity_alloc):
    mu_arr, Sigma = mu.values, cov.values
    def obj(w): return -(w @ mu_arr - 0.5 * A * (w @ Sigma @ w))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
    if max_loss_tolerance:
        z = norm.ppf(1 - var_alpha)
        def var_con(w):
            mean_r = w @ mu_arr
            std_r = math.sqrt(w @ Sigma @ w)
            ann_mean = (1 + mean_r) ** 12 - 1
            ann_std = std_r * math.sqrt(12)
            var_value = -(ann_mean - z * ann_std)
            return max_loss_tolerance - var_value
        cons.append({"type": "ineq", "fun": var_con})
    cons.append({"type": "ineq", "fun": lambda w: max_equity_alloc - np.sum(w[equity_indices])})
    bounds = [(0, 1)] * len(mu)
    res = minimize(obj, np.ones(len(mu)) / len(mu), method="SLSQP", bounds=bounds, constraints=cons)
    return res.x if res.success else None

def compute_integer_shares(weights, prices, total_investment):
    shares_dict = {}
    total_allocated = 0
    for ticker, w in weights.items():
        if ticker == "FIXED_DEPOSIT":
            allocated_amt = total_investment * w
            shares_dict[ticker] = {"Price": np.nan, "Shares": np.nan, "Allocated_Value": allocated_amt, "Allocation_%": w*100}
            total_allocated += allocated_amt
            continue
        price = prices.get(ticker, np.nan)
        if np.isnan(price) or price <= 0:
            shares_dict[ticker] = {"Price": price, "Shares": 0, "Allocated_Value": 0, "Allocation_%": 0}
            continue
        allocated_amt = total_investment * w
        num_shares = int(np.floor(allocated_amt / price))
        actual_allocated = num_shares * price
        total_allocated += actual_allocated
        shares_dict[ticker] = {
            "Price": price,
            "Shares": num_shares,
            "Allocated_Value": actual_allocated,
            "Allocation_%": actual_allocated / total_investment * 100
        }
    leftover_cash = total_investment - total_allocated
    df_shares = pd.DataFrame.from_dict(shares_dict, orient="index")
    df_shares.index.name = "Asset"
    df_shares["Leftover_Cash"] = leftover_cash if "FIXED_DEPOSIT" in df_shares.index else np.nan
    return df_shares

def run_optimizer(user_inputs):
    TOTAL_INVESTMENT = user_inputs["TOTAL_INVESTMENT"]
    TIME_HORIZON_M = user_inputs["TIME_HORIZON_YEARS"] * 12
    MAX_LOSS_TOLERANCE_PCT = user_inputs["RISK_TOLERANCE_LOSS_PCT"]
    INVESTMENT_STYLE = user_inputs["INVESTMENT_STYLE"]
    FD_RETURN_ANNUAL = user_inputs["FD_RATE"] / 100

    MAX_LOSS_TOLERANCE = MAX_LOSS_TOLERANCE_PCT / 100.0

    # Calculate Risk Aversion (A) based on Max Loss Tolerance
    RISK_AVERSION_A = map_risk_aversion_from_var(MAX_LOSS_TOLERANCE)

    # This part of the code now generates the analytics file dynamically.
    generate_portfolio_analytics(TIME_HORIZON_M)
    
    betas, cov, monthly_returns = read_inputs()
    investable_tickers = betas.index.tolist()
    
    # Use FD_RATE as the risk-free rate
    fd_monthly = (1.0 + FD_RETURN_ANNUAL) ** (1.0/12.0) - 1.0

    mu = compute_expected_returns_capm(
        betas=betas,
        monthly_returns=monthly_returns,
        monthly_rf=fd_monthly,
        time_horizon_m=TIME_HORIZON_M
    )
    prices = fetch_latest_prices(betas.index)
    
    # Max Equity Allocation based on Style
    TIME_HORIZON_Y = user_inputs["TIME_HORIZON_YEARS"]
    if INVESTMENT_STYLE.lower() == "aggressive":
        max_equity_alloc = min(0.30 + 0.02 * TIME_HORIZON_Y + 0.35, 0.90)
    elif INVESTMENT_STYLE.lower() == "moderate":
        max_equity_alloc = min(0.30 + 0.02 * TIME_HORIZON_Y + 0.20, 0.90)
    else:
        max_equity_alloc = min(0.30 + 0.02 * TIME_HORIZON_Y, 0.90)

    # Add FD as synthetic asset
    fd_label = "FIXED_DEPOSIT"
    mu_all = pd.concat([mu, pd.Series([fd_monthly], index=[fd_label])])
    prices.loc[fd_label] = np.nan
    cov_all = cov.copy()
    cov_all[fd_label] = 0
    cov_all.loc[fd_label] = 0
    equity_indices = [i for i, lbl in enumerate(mu_all.index) if lbl != fd_label]
    
    w = solve_continuous_utility(
        mu_all, cov_all, RISK_AVERSION_A,
        max_loss_tolerance=MAX_LOSS_TOLERANCE,
        var_alpha=VAR_ALPHA,
        equity_indices=equity_indices,
        max_equity_alloc=max_equity_alloc
    )
    if w is None:
        raise RuntimeError("Optimization failed to find a solution.")

    weights = pd.Series(w, index=mu_all.index)

    port_return = (1 + weights @ mu_all.values) ** 12 - 1
    port_std = math.sqrt(weights @ cov_all.values @ weights) * math.sqrt(12)

    shares_df = compute_integer_shares(weights, prices, TOTAL_INVESTMENT)
    
    summary = pd.DataFrame({
        "Metric": [
            "Total Investment", "Time Horizon (Years)", "Max Loss Tolerance (%)", "Investment Style", "FD Rate (%)", "Risk Aversion (A)"
        ],
        "Value": [
            TOTAL_INVESTMENT, TIME_HORIZON_Y, MAX_LOSS_TOLERANCE_PCT, INVESTMENT_STYLE, FD_RETURN_ANNUAL * 100, RISK_AVERSION_A
        ]
    })
    
    return {
        "summary": summary,
        "shares": shares_df,
        "optimal_weights": weights.to_frame("Weight")
    }

# ==============================================================================
# PART 3: STREAMLIT APP
# ==============================================================================

st.set_page_config(page_title="Intelligent Portfolio Tool", layout="wide")
st.title("Intelligent Portfolio Tool 📈")

st.write("Welcome! Let's build your optimized investment portfolio.")
st.markdown("---")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """
            You are a highly intelligent and friendly financial assistant. Your purpose is to have a natural conversation with the user to understand their investment goals and financial personality. You need to collect five key pieces of information:
            1. Total investment capital (a number, converting 'lakhs' or 'crores' to a standard numerical value).
            2. Investment time horizon in years (a number).
            3. Maximum acceptable loss percentage (a number).
            4. Investment style ('Aggressive', 'Moderate', or 'Conservative').
            5. Expected Fixed Deposit rates (a number).
            
            Based on the conversation, you must fill a JSON object with these values. If a value is unknown, use null.
            Example JSON: {"capital_amount": 2000000, "time_horizon_years": 3, "risk_tolerance_loss_pct": 10, "investment_style": "Moderate", "FD_rate": 6.5}
            
            Your conversation should also include these explanations:
            - Explain what an "expected loss" is, as the maximum potential loss with a 95% probability (also known as VaR at 95%).
            - Explain what the different investment styles depict: Aggressive style implies a high equity allocation, Moderate implies a moderate allocation, and Conservative implies a low allocation.
            
            Start the conversation by greeting the user and asking for their investment details.
            """},
        {"role": "assistant", "content": "Hello! I'm your portfolio assistant. Please tell me about your investment details so I can help you."}
    ]

if "inputs_collected" not in st.session_state:
    st.session_state.inputs_collected = {
        "capital_amount": None,
        "time_horizon_years": None,
        "risk_tolerance_loss_pct": None,
        "investment_style": None,
        "FD_rate": None
    }

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

if not st.session_state.inputs_collected.get("all_found"):
    if prompt := st.chat_input("What are your investment details?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ai_response_text, parsed_data = get_chatbot_response(prompt, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(ai_response_text)
            
        if parsed_data:
            for key in st.session_state.inputs_collected.keys():
                if parsed_data.get(key) is not None:
                    st.session_state.inputs_collected[key] = parsed_data[key]
            
            if None not in st.session_state.inputs_collected.values():
                st.session_state.inputs_collected["all_found"] = True
                st.rerun()

if st.session_state.inputs_collected.get("all_found"):
    st.markdown("---")
    st.subheader("Your Investment Profile Confirmed ✅")
    st.write("We have collected all the necessary details. Please review and confirm below.")
    
    st.json(st.session_state.inputs_collected)
    
    if st.button("Generate Portfolio"):
        with st.spinner("Running optimizer..."):
            user_inputs_for_optimizer = {
                "TOTAL_INVESTMENT": float(st.session_state.inputs_collected["capital_amount"]),
                "TIME_HORIZON_YEARS": float(st.session_state.inputs_collected["time_horizon_years"]),
                "RISK_TOLERANCE_LOSS_PCT": float(st.session_state.inputs_collected["risk_tolerance_loss_pct"]),
                "INVESTMENT_STYLE": st.session_state.inputs_collected["investment_style"],
                "FD_RATE": float(st.session_state.inputs_collected["FD_rate"])
            }
            
            try:
                results = run_optimizer(user_inputs_for_optimizer)
                st.session_state.optimizer_results = results
                st.success("Portfolio generated successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")

if "optimizer_results" in st.session_state:
    st.subheader("Final Portfolio Allocation 📊")
    
    st.write("### Portfolio Summary")
    st.dataframe(st.session_state.optimizer_results["summary"])
    
    st.write("### Recommended Integer Shares")
    st.dataframe(st.session_state.optimizer_results["shares"])
    
    st.write("### Optimal Continuous Weights")
    st.dataframe(st.session_state.optimizer_results["optimal_weights"])
