from datetime import datetime
import os

from dotenv import load_dotenv
import streamlit as st

from agents import UserProfile
from financial_agent import FinancialAgent

st.title("Agentic AI Financial Assistant")
net_worth = st.number_input("Enter your net worth (USD):", min_value=0.0, step=100.0)
risk_tolerance = st.slider("Risk Tolerance (1 = Low, 10 = High):", min_value=1, max_value=10)
goal = st.selectbox("Select your investment goal:", options=["Growth", "Aggressive"])

st.subheader("Current Portfolio")
portfolio = []
num_stocks = st.number_input("How many stocks in your current portfolio?", min_value=1, max_value=20, step=1)
total_weight = 0.0
for i in range(int(num_stocks)):
    col1, col2 = st.columns(2)
    with col1:
        stock = st.text_input(f"Stock #{i+1} Symbol", key=f"stock_{i}")
    with col2:
        weight = st.number_input(f"Weight of {stock} (0-1)", min_value=0.0, max_value=1.0, step=0.01, key=f"weight_{i}")

    if stock:
        portfolio.append((stock.upper(), weight))
        total_weight += weight
if abs(total_weight - 1.0) > 0.001:
    st.warning(f"⚠️ Total weights must sum to 1. Current total: {round(total_weight, 4)}")

# Final Submission
if st.button("Submit"):
    if abs(total_weight - 1.0) > 0.001:
        st.error("Portfolio weights must add up to exactly 1.")
    else:
        st.success("Submitted successfully!")
        user_profile = UserProfile(
            net_worth=net_worth,
            risk_tolerance=risk_tolerance,
            goal=goal,
            current_portfolio=dict(portfolio)
        )
        load_dotenv()
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

        agent = FinancialAgent(user_profile, OPENAI_API_KEY)
        st.write("### Summary of Inputs")
        st.write(f"**Net Worth:** ${net_worth:,.2f}")
        st.write(f"**Risk Tolerance:** {risk_tolerance}")
        st.write(f"**Goal:** {goal}")
        st.write("**Current Portfolio:**")
        st.table(portfolio)

        symbols = list(user_profile.current_portfolio.keys())
        dt = datetime.today()
        for symbol in symbols:
            st.write(symbol, '|', agent.evaluate(symbol, dt))
