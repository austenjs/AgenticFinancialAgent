import numpy as np
import yfinance as yf

from agents.base_agent import BaseAgent, UserProfile

class TechnicalAgent(BaseAgent):
    def __init__(self, user_profile: UserProfile):
        super().__init__()
        self.user_profile = user_profile

    def evaluate(self, symbol, dt):
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='6mo')
        score = 0
        reasons = []

        if 'Close' in hist.columns and len(hist) >= 30:
            closes = hist['Close']
            ma_10 = closes.iloc[-10:].mean()
            ma_20 = closes.iloc[-20:].mean()
            ma_30 = closes.iloc[-30:].mean()
            curr = closes.iloc[-1]

            if curr > ma_10 > ma_20 > ma_30:
                score += 0.15
                reasons.append("All major MAs stacked bullish")
            elif curr < ma_10 < ma_20 < ma_30:
                score -= 0.15
                reasons.append("All major MAs bearish")

            # RSI calculation
            diffs = closes.diff()
            up = diffs.clip(lower=0).rolling(window=14).mean()
            down = -diffs.clip(upper=0).rolling(window=14).mean()
            with np.errstate(divide='ignore', invalid='ignore'):
                rs = up / down
                rsi = 100 - (100 / (1 + rs.iloc[-1]))
            if rsi < 30:
                score += 0.15
                reasons.append("RSI oversold")
            elif rsi > 70:
                score -= 0.15
                reasons.append("RSI overbought")
        else:
            reasons.append("Not enough historical data")

        return {
            "score": score,
            "reasoning": "; ".join(reasons) if reasons else "No strong technical signal"
        }
