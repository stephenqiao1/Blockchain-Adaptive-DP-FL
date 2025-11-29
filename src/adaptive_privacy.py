# adaptive_privacy.py
import torch
import numpy as np
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

class AdaptivePrivacyController:
    def __init__(self, model, dataloader, base_epsilon_target, epochs, strategy='static'):
        """
        strategy: 'static' or 'adaptive'
        """
        self.model = model
        self.dataloader = dataloader
        self.strategy = strategy
        self.epochs = epochs
        
        # Initial Parameters
        self.noise_multiplier = 1.0
        self.max_grad_norm = 1.0
        
        # Adaptive State
        self.prev_loss = float('inf')
        self.patience_counter = 0
        
        # Initialize Opacus
        self.privacy_engine = PrivacyEngine()
        # We manually manage the optimizer later, so we just prep the accountant here
        self.accountant = RDPAccountant()
        
    def adapt_parameters(self, val_loss):
        if self.strategy == 'static':
            return self.noise_multiplier, self.max_grad_norm

        # --- ADAPTIVE LOGIC ---
        # 1. Noise Adaptation:
        # If loss is decreasing (learning well), reduce noise to fine-tune.
        # If loss plateaus/increases, increase noise or keep high to prevent overfitting/memorization.
        if val_loss < self.prev_loss:
            # Improvement detected: Drop noise slightly (e.g., by 5%)
            self.noise_multiplier = max(0.5, self.noise_multiplier * 0.95)
            self.patience_counter = 0
        else:
            # Stagnation: Increase noise (e.g., by 5%)
            self.noise_multiplier = min(2.0, self.noise_multiplier * 1.05)
            self.patience_counter += 1

        # 2. Clipping Adaptation (Optional advanced feature):
        # If gradients are exploding (loss spike), tighten the clip
        if val_loss > self.prev_loss * 1.1: 
             self.max_grad_norm = max(0.5, self.max_grad_norm * 0.9)
        
        self.prev_loss = val_loss
        return self.noise_multiplier, self.max_grad_norm

    def get_privacy_spent(self, delta=1e-5):
        # Calculate epsilon based on history of (noise, sample_rate, steps)
        # Note: In a real advanced implementation, you must track history of distinct sigmas.
        # For this project scope, we query the Opacus accountant.
        return self.accountant.get_epsilon(delta=delta)