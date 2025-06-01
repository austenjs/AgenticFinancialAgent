import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import yfinance as yf

from agents.base_agent import BaseAgent, UserProfile 

class MLPredictorAgent(BaseAgent):
    def __init__(self, user_profile: UserProfile):
        super().__init__()
        self.user_profile = user_profile
        # Fitted models will be cached for performance during multiple .evaluate() calls
        self.models = {}

    def evaluate(self, symbol, dt):
        # 1 month of data, predict next day direction by last 14 days features (returns + volatility)
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period="6mo").dropna()
        if len(hist) < 35:
            return {"score": 0, "reasoning": "Not enough data for ML prediction"}
        closes = hist['Close'].values
        X, y = [], []
        for i in range(21, len(closes)-1):
            window = closes[i-14:i]
            returns_feat = (window[1:] - window[:-1]) / window[:-1]
            vol_feat = np.std(window)
            features = np.concatenate([returns_feat, [vol_feat]])
            X.append(features)
            # Predict UP (+1) if next day's close is higher than today's
            y.append(1 if closes[i+1] > closes[i] else 0)
        X, y = np.array(X), np.array(y)
        scaler = StandardScaler().fit(X)
        X_scaled = scaler.transform(X)
        model = LogisticRegression().fit(X_scaled, y)
        pred_proba = model.predict_proba(scaler.transform(X_scaled[-5:]))[:,1].mean()
        score = (pred_proba - 0.5)  # -0.5 ... +0.5
        # Use score range -0.5,0.5 scaled to -0.15...+0.15 for consistency
        score = score * 0.3
        return {
            "score": score,
            "reasoning": f"MLProba(up)={pred_proba:.2f}"
        }
