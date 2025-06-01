from agents.base_agent import BaseAgent, UserProfile

class MetaDecisionAgent(BaseAgent):
    def __init__(self, user_profile: UserProfile, agents, weights=None):
        super().__init__()
        self.user_profile = user_profile
        self.agents = agents
        self.weights = weights or [1/len(agents)] * len(agents)

    def evaluate(self, symbol, dt):
        subscores = []
        explain = []
        for agent, weight in zip(self.agents, self.weights):
            result = agent.evaluate(symbol, dt)
            subscores.append(weight * result['score'])
            explain.append(f"{agent.__class__.__name__}: {result['reasoning']} ({result['score']:.2f}, weight {weight:.2f})")
        total = sum(subscores)
        rt = self.user_profile.risk_tolerance
        adjusted_total = total if rt > 5 else total * 0.7
        if adjusted_total > 0.15:
            rec = "BUY"
        elif adjusted_total < -0.15:
            rec = "SELL"
        else:
            rec = "HOLD"
        return {
            "symbol": symbol,
            "recommendation": rec,
            "combined_score": adjusted_total,
            "details": " | ".join(explain)
        }
