import yfinance as yf

from agents.base_agent import BaseAgent, UserProfile

class FundamentalAgent(BaseAgent):
    def __init__(self, user_profile: UserProfile):
        super().__init__()
        self.user_profile = user_profile

    def evaluate(self, symbol, dt):
        ticker = yf.Ticker(symbol)
        info = ticker.info
        score = 0
        reasons = []
        pe = info.get('trailingPE', None)
        eps_growth = info.get('earningsQuarterlyGrowth', None)
        roe = info.get('returnOnEquity', None)
        debt_to_eq = info.get('debtToEquity', None)

        # Value: lower P/E is better, but avoid very low (may signal distress)
        if pe is not None:
            if 10 < pe < 20:
                score += 0.15
                reasons.append(f"Attractive P/E ({pe:.1f})")
            elif pe < 8 or pe > 30:
                score -= 0.15
                reasons.append(f"Outlier P/E ({pe:.1f})")

        # Growth: positive EPS growth is good
        if eps_growth is not None:
            if eps_growth > 0.1:
                score += 0.15
                reasons.append(f"Strong earnings growth ({eps_growth:.2%})")
            elif eps_growth < 0:
                score -= 0.15
                reasons.append("Earnings declining")

        # Quality: high ROE and low Debt/Equity is good
        if roe is not None and roe > 0.12:
            score += 0.15
            reasons.append(f"Good ROE ({roe:.2f})")
        if debt_to_eq is not None and debt_to_eq > 150:
            score -= 0.15
            reasons.append(f"High Debt/Equity ({debt_to_eq:.1f})")

        return {
            "score": score,
            "reasoning": "; ".join(reasons) if reasons else "No strong fundamental signal"
        }
