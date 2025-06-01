from datetime import datetime
import os

from dotenv import load_dotenv

from agents import (
    FundamentalAgent,
    TechnicalAgent,
    MetaDecisionAgent,
    MLPredictorAgent,
    NewsSentimentAgent,
    UserProfile
)

class FinancialAgent:
    def __init__(self, user_profile: UserProfile, openai_api_key:str):
        self.news_agent = NewsSentimentAgent(user_profile, openai_api_key, use_llm=True)
        self.fund_agent = FundamentalAgent(user_profile)
        self.tech_agent = TechnicalAgent(user_profile)
        self.ml_agent = MLPredictorAgent(user_profile)
        self.meta_agent = MetaDecisionAgent(
            user_profile,
            [self.news_agent, self.fund_agent, self.tech_agent, self.ml_agent],
            weights = [0.3, 0.3, 0.1, 0.3]
        )
    
    def evaluate(self, symbol, dt):
        return self.meta_agent.evaluate(symbol, dt)

if __name__ == '__main__':
    user_profile = UserProfile(
        net_worth=1000,
        risk_tolerance=5,
        goal="Growth",
        current_portfolio={'AAPL': 0.3, 'TSLA': 0.2, 'MSFT': 0.3, 'NVDA': 0.2}
    )
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") 
    backend = FinancialAgent(user_profile, OPENAI_API_KEY)

    dt = datetime(2025, 5, 31)
    for symbol in user_profile.current_portfolio:
        decision = backend.evaluate(symbol, dt)
        print(f"{symbol}: {decision['recommendation']} (Score: {decision['combined_score']:.2f})")
        print(f"Details: {decision['details']}")
        print("-" * 60)
