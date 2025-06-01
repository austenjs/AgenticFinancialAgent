class BaseAgent:
    def evaluate(self, symbol):
        """Returns dict with at least {'score': ..., 'reasoning': ...}"""
        raise NotImplementedError

class UserProfile:
    def __init__(self, net_worth, risk_tolerance, goal, current_portfolio):
        self.net_worth = net_worth
        self.risk_tolerance = risk_tolerance
        self.goal = goal
        self.current_portfolio = current_portfolio
