import streamlit as st
import os
import sys
import json
import math
import warnings
import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import yfinance as yf

# --- API KEY & SETUP ---
# Fetch the API key from environment variables
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# --- Chatbot-specific state variables ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "inputs_collected" not in st.session_state:
    st.session_state.inputs_collected = {
        "total_investment": None,
        "time_horizon_y": None,
        "max_loss_pct": None,
        "investment_style": None,
        "expected_fd_rate": None,
    }
if "optimizer_ready" not in st.session_state:
    st.session_state.optimizer_ready = False
if "conversation_complete" not in st.session_state:
    st.session_state.conversation_complete = False

# -----------------------------
# PART 1: Backend Optimizer (from THE FINAL DAY.py)
# -----------------------------

# All functions from the original THE FINAL DAY.py are included here.
# NOTE: The original script's "Analytics.py" part is assumed to have run
# and generated the necessary Excel file. In a real-world app, you would
# integrate the analytics part or ensure the file exists. For this tool,
# we are assuming `Nifty50_Portfolio_Analytics.xlsx` is available.

EXCEL_INPUT = "Nifty50_Portfolio_Analytics.xlsx"
EXCEL_OUTPUT = "Portfolio_Optimization_Output.xlsx"
RFR_TICKER = "^IRX"
FALLBACK_ANNUAL_RF = 0.045
DROP_ASSETS_CONTAINING = ["NIFTY_50", "TBILL"]
PRICE_LOOKBACK_DAYS = 5
VAR_ALPHA = 0.95

def read_inputs():
    if not os.path.exists(EXCEL_INPUT):
        st.error(f"Could not find '{EXCEL_INPUT}'. Please ensure the analytics file is in the same directory.")
        st.stop()
    xl = pd.ExcelFile(EXCEL_INPUT)
    betas = pd.read_excel(xl, "Betas", index_col=0)
    cov = pd.read_excel(xl, "Covariance_Matrix", index_col=0)
    monthly_returns = None
    if "Monthly_Returns" in xl.sheet_names:
        monthly_returns = pd.read_excel(xl, "Monthly_Returns", index_col=0, parse_dates=True)

    cov = cov.loc[cov.index.intersection(cov.columns), cov.columns.intersection(cov.index)]
    assets = list(cov.columns)
    investable = [a for a in assets if not any(tag in a.upper() for tag in DROP_ASSETS_CONTAINING)]
    beta_series = betas.squeeze().reindex(investable).astype(float)
    cov_invest = cov.loc[investable, investable]
    return beta_series, cov_invest, monthly_returns

def fetch_latest_risk_free_monthly():
    annual_rf = None
    try:
        r = yf.download(RFR_TICKER, period="1mo", interval="1d", progress=False)
        if not r.empty and "Close" in r:
            latest = r["Close"].dropna().iloc[-1]
            annual_rf = (latest / 100.0) if latest > 1 else float(latest)
    except Exception:
        pass
    if annual_rf is None:
        annual_rf = FALLBACK_ANNUAL_RF
    monthly_rf = (1 + annual_rf) ** (1 / 12) - 1
    return monthly_rf, annual_rf

def compute_expected_returns_capm(betas, monthly_rf, monthly_returns, horizon_m):
    if monthly_returns is not None and "NIFTY_50" in monthly_returns.columns:
        mret = monthly_returns["NIFTY_50"].dropna().tail(horizon_m)
        exp_rm = mret.mean() if len(mret) > 0 else monthly_rf + 0.0065
    else:
        exp_rm = monthly_rf + 0.0065
    exp_excess = exp_rm - monthly_rf
    return pd.Series(monthly_rf + betas.values * exp_excess, index=betas.index)

def fetch_latest_prices(tickers):
    prices = {}
    for t in tickers:
        try:
            hist = yf.download(t, period=f"{PRICE_LOOKBACK_DAYS}d", interval="1d", progress=False)["Close"].dropna()
            prices[t] = float(hist.iloc[-1]) if not hist.empty else np.nan
        except:
            prices[t] = np.nan
    return pd.Series(prices)

def map_risk_aversion_from_var(MAX_LOSS_TOLERANCE):
    if MAX_LOSS_TOLERANCE >= 0.50: return 1
    elif MAX_LOSS_TOLERANCE >= 0.40: return 2
    elif MAX_LOSS_TOLERANCE >= 0.35: return 3
    elif MAX_LOSS_TOLERANCE >= 0.30: return 4
    elif MAX_LOSS_TOLERANCE >= 0.25: return 5
    elif MAX_LOSS_TOLERANCE >= 0.20: return 6
    elif MAX_LOSS_TOLERANCE >= 0.15: return 7
    elif MAX_LOSS_TOLERANCE >= 0.10: return 8
    elif MAX_LOSS_TOLERANCE >= 0.05: return 9
    else: return 10

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
            shares_dict[ticker] = {"Price": np.nan, "Shares": np.nan, "Allocated_Value": allocated_amt, "Allocation_%": w * 100}
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
    TOTAL_INVESTMENT = user_inputs["total_investment"]
    TIME_HORIZON_Y = user_inputs["time_horizon_y"]
    MAX_LOSS_TOLERANCE = user_inputs["max_loss_pct"] / 100
    FD_RETURN_ANNUAL = user_inputs["expected_fd_rate"] / 100
    STYLE = user_inputs["investment_style"]

    time_horizon_m = int(round(TIME_HORIZON_Y * 12))
    
    betas, cov, monthly_returns = read_inputs()
    rf_monthly, rf_annual = fetch_latest_risk_free_monthly()
    mu_equities = compute_expected_returns_capm(betas, rf_monthly, monthly_returns, time_horizon_m)
    prices = fetch_latest_prices(betas.index)

    fd_monthly = (1 + FD_RETURN_ANNUAL) ** (1 / 12) - 1

    if STYLE == "Aggressive":
        max_equity_alloc = min(0.30 + 0.02 * TIME_HORIZON_Y + 0.35, 0.90)
    elif STYLE == "Moderate":
        max_equity_alloc = min(0.30 + 0.02 * TIME_HORIZON_Y + 0.20, 0.90)
    else: # Conservative
        max_equity_alloc = min(0.30 + 0.02 * TIME_HORIZON_Y, 0.90)

    fd_label = "FIXED_DEPOSIT"
    mu_all = mu_equities.copy()
    mu_all.loc[fd_label] = fd_monthly
    cov_all = cov.copy()
    cov_all[fd_label] = 0
    cov_all.loc[fd_label] = 0
    equity_indices = [i for i, lbl in enumerate(mu_all.index) if lbl != fd_label]
    
    w_eq = np.ones(len(mu_equities)) / len(mu_equities)
    mean_r = w_eq @ mu_equities.values
    std_r = math.sqrt(w_eq @ cov.values @ w_eq)
    ann_mean = (1 + mean_r) ** 12 - 1
    ann_std = std_r * math.sqrt(12)
    z = norm.ppf(1 - VAR_ALPHA)
    var_value = -(ann_mean - z * ann_std)
    A = map_risk_aversion_from_var(MAX_LOSS_TOLERANCE)

    w = solve_continuous_utility(mu_all, cov_all, A, MAX_LOSS_TOLERANCE, VAR_ALPHA, equity_indices, max_equity_alloc)
    if w is None:
        return "Solver failed. Please try different inputs."
    
    weights = pd.Series(w, index=mu_all.index)
    port_return = (1 + weights @ mu_all.values) ** 12 - 1
    port_std = math.sqrt(weights @ cov_all.values @ weights) * math.sqrt(12)
    shares_df = compute_integer_shares(weights, prices, TOTAL_INVESTMENT)

    return {
        "weights": weights,
        "return": port_return,
        "risk": port_std,
        "shares": shares_df,
    }

# -----------------------------
# PART 2: Streamlit & Chatbot
# -----------------------------

def initialize_chat():
    """Initial greeting message from the assistant."""
    st.session_state.messages.append(
        {"role": "assistant", "content": "Hello! I am your personal portfolio optimization assistant. I can help you create a personalized investment plan. Let's start with a few questions. To get started, what is the total investment amount you have in mind?"}
    )

def handle_user_input(user_prompt):
    """
    Processes user input to collect the 5 required values.
    Uses simple keyword/context matching and type conversion.
    """
    user_prompt_lower = user_prompt.lower()
    
    if st.session_state.inputs_collected["total_investment"] is None:
        try:
            amount = float("".join(c for c in user_prompt if c.isdigit() or c == "."))
            if amount > 0:
                st.session_state.inputs_collected["total_investment"] = amount
                return "Thank you. Now, what is your investment time horizon in years?"
            else:
                return "Please provide a valid investment amount greater than zero."
        except (ValueError, IndexError):
            return "I couldn't understand that. Please provide the total investment amount as a number."

    if st.session_state.inputs_collected["time_horizon_y"] is None:
        try:
            years = float("".join(c for c in user_prompt if c.isdigit() or c == "."))
            if years > 0:
                st.session_state.inputs_collected["time_horizon_y"] = years
                return "Great. Before we talk about risk, let's clarify something. VaR (Value at Risk) is a way to measure the **maximum expected loss** for a portfolio over a specific time period. For our model, we'll use a 95% confidence level. What is the maximum percentage you're willing to lose?"
            else:
                return "Please provide a valid time horizon in years greater than zero."
        except (ValueError, IndexError):
            return "I couldn't understand that. Please provide the time horizon as a number of years."

    if st.session_state.inputs_collected["max_loss_pct"] is None:
        try:
            loss = float("".join(c for c in user_prompt if c.isdigit() or c == "."))
            if 0 <= loss <= 100:
                st.session_state.inputs_collected["max_loss_pct"] = loss
                return "Got it. Your investment style also helps determine the portfolio mix. A **Conservative** style aims for stability with lower risk and return. A **Moderate** style balances growth and safety. An **Aggressive** style prioritizes high growth with higher risk. What is your preferred investment style?"
            else:
                return "Please provide a percentage between 0 and 100."
        except (ValueError, IndexError):
            return "I couldn't understand that. Please provide the maximum acceptable loss as a percentage (e.g., 10 for 10%)."

    if st.session_state.inputs_collected["investment_style"] is None:
        if "aggressive" in user_prompt_lower:
            st.session_state.inputs_collected["investment_style"] = "Aggressive"
            return "Aggressive style selected. Finally, what is the current expected annual return on a Fixed Deposit (as a percentage, e.g., 6.5)?"
        elif "moderate" in user_prompt_lower:
            st.session_state.inputs_collected["investment_style"] = "Moderate"
            return "Moderate style selected. Finally, what is the current expected annual return on a Fixed Deposit (as a percentage, e.g., 6.5)?"
        elif "conservative" in user_prompt_lower:
            st.session_state.inputs_collected["investment_style"] = "Conservative"
            return "Conservative style selected. Finally, what is the current expected annual return on a Fixed Deposit (as a percentage, e.g., 6.5)?"
        else:
            return "Please choose one of the three styles: Aggressive, Moderate, or Conservative."

    if st.session_state.inputs_collected["expected_fd_rate"] is None:
        try:
            fd_rate = float("".join(c for c in user_prompt if c.isdigit() or c == "."))
            if fd_rate >= 0:
                st.session_state.inputs_collected["expected_fd_rate"] = fd_rate
                st.session_state.optimizer_ready = True
                return "Thank you! I have all the information I need. You can now click the 'Generate Portfolio' button to see your optimized portfolio."
            else:
                return "Please provide a valid FD rate."
        except (ValueError, IndexError):
            return "I couldn't understand that. Please provide the FD rate as a percentage."
    
    return "I have all the information needed. Please click the 'Generate Portfolio' button."

# --- Streamlit UI ---
st.title("💰 Portfolio Optimization Chatbot")
st.markdown("---")

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Initial greeting message
if not st.session_state.messages:
    initialize_chat()

# Main chat input loop
if prompt := st.chat_input("Start a conversation here..."):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Process the prompt to get the next bot message
    response = handle_user_input(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Button to generate portfolio when all inputs are collected
if st.session_state.optimizer_ready:
    if st.button("Generate Portfolio", type="primary"):
        with st.spinner("Optimizing your portfolio... This may take a moment."):
            try:
                # Call the main optimizer function
                results = run_optimizer(st.session_state.inputs_collected)
                if isinstance(results, str):
                    st.error(results) # Display solver failure
                else:
                    st.session_state.results = results
                    st.session_state.conversation_complete = True
            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")
            finally:
                st.experimental_rerun()

# Display results if optimization is complete
if st.session_state.conversation_complete:
    results = st.session_state.results
    st.markdown("---")
    st.subheader("✅ Optimized Portfolio Results")
    st.markdown("### Portfolio Summary")
    st.metric(label="Expected Annual Return", value=f"{results['return']:.2%}")
    st.metric(label="Expected Annual Risk (Standard Deviation)", value=f"{results['risk']:.2%}")

    st.markdown("### Recommended Asset Weights")
    st.dataframe(results['weights'].to_frame("Weight").style.format({"Weight": "{:.2%}"}))

    st.markdown("### Equity-wise Shares Allocation")
    st.dataframe(results['shares'].style.format({
        "Price": "₹{:,.2f}",
        "Shares": "{:,.0f}",
        "Allocated_Value": "₹{:,.2f}",
        "Allocation_%": "{:.2f}%",
        "Leftover_Cash": "₹{:,.2f}"
    }))
