import os

from dotenv import load_dotenv
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf

from agents import UserProfile
from financial_agent import FinancialAgent

if __name__ == '__main__':
    user_profile = UserProfile(
        net_worth=10000,
        risk_tolerance=5,
        goal="Growth",
        current_portfolio={'AAPL': 0.3, 'TSLA': 0.2, 'MSFT': 0.3, 'NVDA': 0.2}
    )
    symbols = list(user_profile.current_portfolio.keys())
    start_date = "2025-05-15"
    end_date = "2025-05-31"

    # --- Download price data ---
    prices = yf.download(symbols, start=start_date, end=end_date, interval='1d')['Close'].dropna()
    if isinstance(prices, pd.Series):
        prices = prices.to_frame()

    portfolio_values = []
    cash = user_profile.net_worth
    holdings = {sym: 0 for sym in symbols}

    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    ai_agent = FinancialAgent(user_profile, openai_api_key=OPENAI_API_KEY)

    # --- Daily rebalancing backtest ---
    for dt, price_row in prices.iterrows():
        dt = dt.to_pydatetime()
        recs = {sym: ai_agent.evaluate(sym, dt)['recommendation'] for sym in symbols}
        buy_syms = [sym for sym, rec in recs.items() if rec == 'BUY']
        sell_syms = [sym for sym, rec in recs.items() if rec == 'SELL']
        hold_syms = [sym for sym, rec in recs.items() if rec == 'HOLD']
        print(f"Day: {dt}, Buy signals: {buy_syms}, Sell signals: {sell_syms}, Hold signals: {hold_syms}, Cash: {cash}")

        # Sell first (free up cash)
        for sym in sell_syms:
            if holdings[sym] > 0:
                cash += holdings[sym] * price_row[sym]
                holdings[sym] = 0

        # Allocate cash to BUY (honor user_profile.current_portfolio weights)
        if buy_syms:
            # Use weights from user_profile, scaled to just the BUY signals
            weight_sum = sum(user_profile.current_portfolio[sym] for sym in buy_syms)
            for sym in buy_syms:
                target_weight = user_profile.current_portfolio[sym] / weight_sum if weight_sum else 1.0/len(buy_syms)
                to_invest = cash * target_weight
                shares_to_buy = to_invest // price_row[sym]
                cash -= shares_to_buy * price_row[sym]
                holdings[sym] += shares_to_buy

        # Calculate daily portfolio value
        value = cash + sum(holdings[sym] * price_row[sym] for sym in symbols)
        portfolio_values.append(value)

    # --- PERFORMANCE METRICS ---
    portfolio_values = pd.Series(portfolio_values, index=prices.index)
    returns = portfolio_values.pct_change().dropna()
    total_return = (portfolio_values.iloc[-1] - user_profile.net_worth) / user_profile.net_worth
    sharpe = returns.mean() / returns.std() * (252 ** 0.5)
    max_drawdown = ((portfolio_values.cummax() - portfolio_values) / portfolio_values.cummax()).max()

    print(f"Final Portfolio Value: ${portfolio_values.iloc[-1]:.2f}")
    print(f"Total Return: {total_return*100:.2f}%")
    print(f"Sharpe Ratio: {sharpe:.2f}")
    print(f"Max Drawdown: {max_drawdown*100:.2f}%")

    # --- PLOT AND SAVE ---
    portfolio_values.plot(label='Agent Portfolio')
    plt.title('Portfolio Value Over Time (Daily Rebalancing)')
    plt.ylabel('Portfolio Value ($)')
    plt.xlabel('Date')
    plt.legend()
    plt.savefig('backtest_plot.jpg')
