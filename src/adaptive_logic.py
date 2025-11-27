import numpy as np

class AdaptivePrivacyScaler:
    def __init__(self, base_sigma=1.0, min_sigma=0.5, max_sigma=2.0):
        self.sigma = base_sigma
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.alpha = 0.8

    def adjust_sigma(self, val_loss, val_acc):
        """
        Adjusts noise multiplier (sigma) based on model performance.
        Logic:
        - If loss is high (>1.5), we need clarity -> reduce sigma (spend more budget)
        - If loss is low (<0.5), we are safe -> increase sigma (save budget)
        """

        target_sigma = self.sigma
        
        if val_loss > 1.5:
            # Hard to learn: Decrease noise to see data clearer
            target_sigma = max(self.min_sigma, self.sigma * 0.9)
        elif val_loss < 0.5:
            # Easy to learn: Increase noise to save privacy budget
            target_sigma = min(self.max_sigma, self.sigma * 1.1)

        # Smooth update
        self.sigma = self.alpha * self.sigma + (1 - self.alpha) * target_sigma
        return self.sigma
    
    def calculate_epsilon_cost(self, sigma, steps=1):
        """
        Approximation of privacy cost for this round.
        Lower sigma -> higher epsilon cost.
        """
        base_cost = 1.0
        round_cost = base_cost / sigma
        return round_cost