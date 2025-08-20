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
            Example JSON: {"capital_amount": 2000000, "time_horizon_years": 3, "risk_tolerance_loss_pct": 10, "risk_aversion_A": 5.0}
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
# PART 2: PORTFOLIO OPTIMIZER (UNCHANGED)
# ==============================================================================

EXCEL_INPUT = "Nifty50_Portfolio_Analytics.xlsx"
SHEET_BETAS = "Betas"
SHEET_COV = "Covariance_Matrix"
SHEET_RET = "Monthly_Returns"
RFR_TICKER = "^IRX"
FALLBACK_ANNUAL_RF = 0.045
DROP_ASSETS_CONTAINING = ["NIFTY_50", "TBILL"]
PRICE_LOOKBACK_DAYS = 5
EXCEL_OUTPUT = "Portfolio_Optimization_Output.xlsx"

def read_inputs():
    if not os.path.exists(EXCEL_INPUT):
        raise FileNotFoundError(f"Could not find '{EXCEL_INPUT}' in current folder.")
    xl = pd.ExcelFile(EXCEL_INPUT)
    betas = pd.read_excel(xl, SHEET_BETAS, index_col=0)
    cov = pd.read_excel(xl, SHEET_COV, index_col=0)
    monthly_returns = None
    if SHEET_RET in xl.sheet_names:
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

def fetch_latest_risk_free_monthly():
    if RFR_TICKER is None and FALLBACK_ANNUAL_RF is None:
        raise ValueError("Provide RFR_TICKER or FALLBACK_ANNUAL_RF")
    annual_rf = None
    if RFR_TICKER:
        try:
            r = yf.download(RFR_TICKER, period="1mo", interval="1d", progress=False)
            if not r.empty and "Close" in r:
                latest = r["Close"].dropna().iloc[-1]
                annual_rf = (latest / 100.0) if latest > 1 else float(latest)
        except Exception as e:
            warnings.warn(f"Risk-free fetch failed: {e}")
    if annual_rf is None:
        if FALLBACK_ANNUAL_RF is None:
            raise RuntimeError("Could not fetch risk-free and no FALLBACK_ANNUAL_RF set.")
        annual_rf = float(FALLBACK_ANNUAL_RF)
    monthly_rf = (1.0 + annual_rf) ** (1.0/12.0) - 1.0
    return monthly_rf, annual_rf

def compute_expected_returns_capm(betas, monthly_rf, monthly_returns, time_horizon_m):
    if monthly_returns is not None and "NIFTY_50" in monthly_returns.columns:
        mret = monthly_returns["NIFTY_50"].dropna()
        if time_horizon_m is not None and time_horizon_m > 0:
            mret = mret.tail(time_horizon_m)
        if len(mret) == 0:
            raise ValueError("No NIFTY_50 returns found to compute market return.")
        exp_rm = mret.mean()
    else:
        warnings.warn("Monthly_Returns with NIFTY_50 not found; assuming market premium of 0.5%/month.")
        exp_rm = monthly_rf + 0.0065
    exp_excess = exp_rm - monthly_rf
    exp_rets = monthly_rf + betas.values * exp_excess
    return pd.Series(exp_rets, index=betas.index, name="E_monthly_CAPM")

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

def solve_continuous_utility(mu, cov, A, max_loss_tolerance, var_alpha):
    n = len(mu)
    mu = np.asarray(mu, dtype=float)
    Sigma = np.asarray(cov, dtype=float)
    def obj(w):
        return -(w @ mu - 0.5 * A * (w @ Sigma @ w))
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    if max_loss_tolerance is not None:
        z = norm.ppf(1 - var_alpha)
        def var_con(w):
            mean_r = w @ mu
            std_r  = math.sqrt(w @ Sigma @ w)
            ann_mean = (1 + mean_r)**12 - 1
            ann_std  = std_r * math.sqrt(12)
            var_95 = -(ann_mean + z * ann_std)
            return max_loss_tolerance - var_95
        cons.append({"type": "ineq", "fun": var_con})
    bounds = [(0.0, 1.0)] * n
    w0 = np.ones(n) / n
    res = minimize(obj, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200})
    if not res.success:
        warnings.warn(f"Continuous optimizer warning: {res.message}")
    w = np.clip(res.x, 0, 1)
    w = w / w.sum()
    util = w @ mu - 0.5 * A * (w @ Sigma @ w)
    std = math.sqrt(w @ Sigma @ w)
    return w, util, std

def integer_shares_from_weights(weights, prices, total_investment):
    n = {}
    for t in weights.index:
        alloc_value = weights[t] * total_investment
        p = prices[t]
        if pd.isna(p) or p <= 0:
            n[t] = 0
        else:
            n[t] = int(math.floor(alloc_value / p))
    invested_values = prices * pd.Series(n)
    invested_values = invested_values.fillna(0.0)
    invested_total = invested_values.sum()
    realized_weights = invested_values / total_investment
    cash_weight = max(0.0, 1.0 - realized_weights.sum())
    return pd.Series(n, name="Shares"), realized_weights, cash_weight, invested_total

def portfolio_stats(weights, mu, cov, rf_monthly, cash_weight):
    w = weights.fillna(0.0).values
    mu_vec = mu.reindex(weights.index).values
    Sigma = cov.reindex(index=weights.index, columns=weights.index).values
    mean_r = w @ mu_vec
    var_r = w @ Sigma @ w
    if cash_weight > 0:
        mean_r = mean_r + cash_weight * rf_monthly
    std_r = math.sqrt(max(var_r, 0.0))
    return mean_r, std_r

def run_optimizer(user_inputs):
    TOTAL_INVESTMENT = user_inputs['TOTAL_INVESTMENT']
    RISK_AVERSION_A = user_inputs['RISK_AVERSION_A']
    TIME_HORIZON_M = user_inputs['TIME_HORIZON_M']
    MAX_LOSS_TOLERANCE = user_inputs['RISK_TOLERANCE_LOSS_PCT'] / 100.0
    VAR_ALPHA = 0.95
    betas, cov, monthly_returns = read_inputs()
    investable_tickers = betas.index.tolist()
    rf_monthly, rf_annual = fetch_latest_risk_free_monthly()
    mu = compute_expected_returns_capm(
        betas=betas,
        monthly_rf=rf_monthly,
        monthly_returns=monthly_returns,
        time_horizon_m=TIME_HORIZON_M
    )
    prices = fetch_latest_prices(investable_tickers)
    w_cont, util_cont, std_cont = solve_continuous_utility(
        mu, cov, RISK_AVERSION_A,
        max_loss_tolerance=MAX_LOSS_TOLERANCE,
        var_alpha=VAR_ALPHA
    )
    w_cont = pd.Series(w_cont, index=investable_tickers, name="Weight")
    shares, w_realized, cash_w, invested_total = integer_shares_from_weights(w_cont, prices, TOTAL_INVESTMENT)
    er_cont, sd_cont = portfolio_stats(w_cont, mu, cov, rf_monthly=rf_monthly, cash_weight=0.0)
    er_int, sd_int = portfolio_stats(w_realized, mu, cov, rf_monthly=rf_monthly, cash_weight=cash_w)
    util_int = er_int - 0.5 * RISK_AVERSION_A * (sd_int ** 2)
    ann_er_cont = (1 + er_cont)**12 - 1
    ann_sd_cont = sd_cont * math.sqrt(12)
    ann_er_int  = (1 + er_int)**12 - 1
    ann_sd_int  = sd_int * math.sqrt(12)
    exp_returns_df = mu.to_frame()
    prices_df = prices.to_frame()
    w_cont_df = w_cont.to_frame()
    w_realized_df = w_realized.rename("RealizedWeight").to_frame()
    shares_df = shares.to_frame()
    summary = pd.DataFrame({
        "Metric": [
            "Annual Risk-free (input/fetched)",
            "Monthly Risk-free (derived)",
            "Cont. Portfolio E[Rp] (monthly)",
            "Cont. Portfolio Std (monthly)",
            "Cont. Portfolio E[Rp] (annual)",
            "Cont. Portfolio Std (annual)",
            "Cont. Utility",
            "Integer E[Rp] (monthly, inc. cash)",
            "Integer Std (monthly, inc. cash)",
            "Integer E[Rp] (annual, inc. cash)",
            "Integer Std (annual, inc. cash)",
            "Integer Utility",
            "Realized cash weight",
            "Total Invested (from shares)"
        ],
        "Value": [
            rf_annual, rf_monthly, er_cont, sd_cont, ann_er_cont, ann_sd_cont, util_cont, er_int, sd_int, ann_er_int, ann_sd_int, util_int, cash_w, invested_total
        ]
    })
    
    return {
        "summary": summary,
        "shares": shares_df,
        "optimal_weights": w_cont_df
    }

# ==============================================================================
# PART 3: STREAMLIT APP
# ==============================================================================

st.set_page_config(page_title="Intelligent Portfolio Tool", layout="wide")
st.title("Intelligent Portfolio Tool 📈")

st.write("Welcome! Let's build your optimized investment portfolio.")
st.markdown("---")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": """
            You are a highly intelligent and friendly financial assistant. Your purpose is to have a natural conversation with the user to understand their investment goals and financial personality. You need to collect four key pieces of information:
            1. Total investment capital (a number, converting 'lakhs' or 'crores' to a standard numerical value).
            2. Investment time horizon in years (a number).
            3. Maximum acceptable loss percentage (a number).
            4. The user's risk aversion coefficient (A), which you must infer from the user's description of their financial personality or goals (e.g., 'aggressive', 'conservative', 'I can handle a lot of risk'). Do not ask for a number for this.
            
            Based on the conversation, you must fill a JSON object with these values. If a value is unknown, use null.
            Example JSON: {"capital_amount": 2000000, "time_horizon_years": 3, "risk_tolerance_loss_pct": 10, "risk_aversion_A": 5.0}

            Start the conversation by greeting the user and asking for their investment details.
            """},
        {"role": "assistant", "content": "Hello! I'm your portfolio assistant. Please tell me about your investment details so I can help you."}
    ]

if "inputs_collected" not in st.session_state:
    st.session_state.inputs_collected = {
        "capital_amount": None,
        "time_horizon_years": None,
        "risk_tolerance_loss_pct": None,
        "risk_aversion_A": None
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
                "RISK_AVERSION_A": float(st.session_state.inputs_collected["risk_aversion_A"]),
                "TIME_HORIZON_M": int(st.session_state.inputs_collected["time_horizon_years"] * 12),
                "RISK_TOLERANCE_LOSS_PCT": float(st.session_state.inputs_collected["risk_tolerance_loss_pct"])
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