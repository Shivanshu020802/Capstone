import streamlit as st
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

# ==============================================================================
# PART 1: GENAI CONVERSATIONAL INTERFACE
# ==============================================================================

# Global constant for the OpenAI model
# NOTE: The user's provided snippet used 'gpt-4o-mini', which is what we will use.
OPENAI_MODEL = "gpt-4o-mini" 

def get_chatbot_response(user_input, conversation_history):
    """
    Sends user input and conversation history to GenAI to get a response.
    It first gets a conversational response, then a structured JSON.
    """
    try:
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        if not client.api_key:
            st.error("Error: OPENAI_API_KEY environment variable not set.")
            return "Please set your OpenAI API key as an environment variable.", None
    except Exception as e:
        st.error(f"Error initializing OpenAI client: {e}")
        return "An error occurred. Please check your API key and connection.", None

    # Append user input to conversation history for conversational response
    conversation_history.append({"role": "user", "content": user_input})

    # Get conversational AI response
    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=conversation_history,
            temperature=0.7
        )
        ai_response = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_response})
    except Exception as e:
        st.error(f"Error getting AI response: {e}")
        return "An error occurred with the AI. Please try again.", None

    # Get structured JSON response for inputs
    parsing_prompt = conversation_history + [
        {"role": "system", "content": """
            Based on the entire conversation so far, extract the five required values and return them in a JSON object. If a value is still missing, use null.
            Do not include any other text.
            The five values are:
            1. capital_amount (number)
            2. time_horizon_years (number)
            3. risk_tolerance_loss_pct (number)
            4. investment_style (string: 'Aggressive', 'Moderate', or 'Conservative')
            5. expected_fd_rate (number)

            Example JSON: {"capital_amount": 2000000, "time_horizon_years": 3, "risk_tolerance_loss_pct": 10, "investment_style": "Moderate", "expected_fd_rate": 6.5}
            """}
    ]
    try:
        parsing_response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=parsing_prompt,
            response_format={"type": "json_object"},
            temperature=0.0
        )
        parsed_data = json.loads(parsing_response.choices[0].message.content)
        return ai_response, parsed_data
    except Exception as e:
        # If parsing fails, just return the conversational response
        return ai_response, None

# ==============================================================================
# PART 2: ANALYTICS & OPTIMIZER LOGIC (from THE FINAL DAY.py)
# ==============================================================================

# Constants from the original script
EXCEL_INPUT = "Nifty50_AllAssets.xlsx"
ANALYTICS_OUTPUT = "Nifty50_Portfolio_Analytics.xlsx"
OPTIMIZER_OUTPUT = "Portfolio_Optimization_Output.xlsx"
RFR_TICKER = "^IRX"
FALLBACK_ANNUAL_RF = 0.045
DROP_ASSETS_CONTAINING = ["NIFTY_50", "TBILL"]
PRICE_LOOKBACK_DAYS = 5
VAR_ALPHA = 0.95

def generate_portfolio_analytics(time_horizon_input):
    """
    Runs the analytics part of the original script to generate the necessary
    Excel file for the optimizer.
    """
    if not os.path.exists(EXCEL_INPUT):
        st.error(f"Error: The data file '{EXCEL_INPUT}' was not found. Please ensure it's in the same directory.")
        st.stop()

    # --- Load Data ---
    df = pd.read_excel(EXCEL_INPUT, sheet_name="Data", parse_dates=["Date"])
    df.set_index("Date", inplace=True)

    # --- Step 1: Convert TBILL_3M yields into price series ---
    if "TBILL_3M" in df.columns:
        yield_series = df["TBILL_3M"] / 100
        price_tbill = 100 / (1 + yield_series * (91 / 360))
        df["TBILL_3M_Price"] = price_tbill
        df.drop(columns=["TBILL_3M"], inplace=True)

    # --- Step 3: Determine Resample Frequency and Horizon in periods ---
    time_horizon_years_analytics = min(time_horizon_input, 10)
    if 0 < time_horizon_years_analytics <= 2:
        freq = "D"
        horizon_periods = time_horizon_years_analytics * 252
        freq_name = "Daily"
    elif 3 <= time_horizon_years_analytics <= 6:
        freq = "W"
        horizon_periods = time_horizon_years_analytics * 52
        freq_name = "Weekly"
    else:
        freq = "M"
        horizon_periods = time_horizon_years_analytics * 12
        freq_name = "Monthly"

    # --- Step 4: Resample Prices based on chosen frequency ---
    prices = df.resample(freq).last()

    # --- Step 5: Fill Missing Values (Backfill + Forward fill) ---
    prices = prices.bfill().ffill()

    # --- Step 6: Calculate Returns for chosen frequency ---
    returns = prices.pct_change().dropna()

    # --- Step 7: Take last horizon_periods returns ---
    returns_subset = returns.tail(int(horizon_periods))

    # --- Step 8: Calculate Betas vs NIFTY_50 ---
    betas = {}
    nifty_returns = returns_subset["NIFTY_50"]
    for col in returns_subset.columns:
        if col == "NIFTY_50":
            continue
        cov = np.cov(returns_subset[col].dropna(), nifty_returns.dropna())[0][1]
        var = np.var(nifty_returns.dropna())
        beta = cov / var if var != 0 else np.nan
        betas[col] = beta
    beta_df = pd.DataFrame.from_dict(betas, orient="index", columns=[f"Beta_vs_NIFTY50_{freq_name}"])

    # --- Step 9: Covariance Matrix ---
    cov_matrix = returns_subset.cov()

    # --- Step 10: Calculate Monthly Returns (ALWAYS) ---
    monthly_prices = df.resample("M").last()
    monthly_prices = monthly_prices.bfill().ffill()
    monthly_returns = monthly_prices.pct_change().dropna()

    # --- Step 11: Save Results ---
    with pd.ExcelWriter(ANALYTICS_OUTPUT, engine="openpyxl") as writer:
        returns.to_excel(writer, sheet_name=f"{freq_name}_Returns")
        returns_subset.to_excel(writer, sheet_name=f"Returns_Last_{time_horizon_years_analytics}Y")
        monthly_returns.to_excel(writer, sheet_name="Monthly_Returns")
        beta_df.to_excel(writer, sheet_name="Betas")
        cov_matrix.to_excel(writer, sheet_name="Covariance_Matrix")

def read_inputs():
    if not os.path.exists(ANALYTICS_OUTPUT):
        raise FileNotFoundError(f"Could not find '{ANALYTICS_OUTPUT}'. Please run analytics first.")
    xl = pd.ExcelFile(ANALYTICS_OUTPUT)
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
    TOTAL_INVESTMENT = user_inputs["capital_amount"]
    TIME_HORIZON_Y = user_inputs["time_horizon_years"]
    MAX_LOSS_TOLERANCE = user_inputs["risk_tolerance_loss_pct"] / 100
    FD_RETURN_ANNUAL = user_inputs["expected_fd_rate"] / 100
    STYLE = user_inputs["investment_style"]

    time_horizon_m = int(round(TIME_HORIZON_Y * 12))
    
    # Run the analytics part first to ensure the data is up-to-date
    generate_portfolio_analytics(TIME_HORIZON_Y)

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

# ==============================================================================
# PART 3: STREAMLIT APP
# ==============================================================================

st.set_page_config(page_title="Intelligent Portfolio Tool", layout="wide")
st.title("Intelligent Portfolio Tool 📈")
st.markdown("---")

# Initialize chat history in session state
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
            Example JSON: {"capital_amount": 2000000, "time_horizon_years": 3, "risk_tolerance_loss_pct": 10, "investment_style": "Moderate", "expected_fd_rate": 6.5}
            
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
        "expected_fd_rate": None
    }
    st.session_state.all_inputs_collected = False
    st.session_state.results = None

for message in st.session_state.messages:
    if message["role"] != "system":
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Main chat input loop
if not st.session_state.all_inputs_collected:
    if prompt := st.chat_input("What are your investment details?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        ai_response_text, parsed_data = get_chatbot_response(prompt, st.session_state.messages)

        with st.chat_message("assistant"):
            st.markdown(ai_response_text)
            
        if parsed_data:
            # Update inputs from the structured response
            for key in st.session_state.inputs_collected.keys():
                if parsed_data.get(key) is not None:
                    st.session_state.inputs_collected[key] = parsed_data[key]
            
            # Check if all inputs are now collected
            if all(value is not None for value in st.session_state.inputs_collected.values()):
                st.session_state.all_inputs_collected = True
                st.session_state.messages.append({"role": "assistant", "content": "Thank you! I have all the information I need. Please click the 'Generate Portfolio' button to see your optimized portfolio."})
                st.experimental_rerun()

# Button to generate portfolio when all inputs are collected
if st.session_state.all_inputs_collected and st.session_state.results is None:
    st.markdown("---")
    st.subheader("Your Investment Profile Confirmed ✅")
    st.write("We have collected all the necessary details. Please review and confirm below.")
    st.json(st.session_state.inputs_collected)

    if st.button("Generate Portfolio", type="primary"):
        with st.spinner("Optimizing your portfolio... This may take a moment."):
            try:
                results = run_optimizer(st.session_state.inputs_collected)
                if isinstance(results, str):
                    st.error(results)
                else:
                    st.session_state.results = results
            except Exception as e:
                st.error(f"An error occurred during optimization: {e}")
            finally:
                pass # Let the next block handle the display

# Display results if optimization is not complete
if st.session_state.results is not None:
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
