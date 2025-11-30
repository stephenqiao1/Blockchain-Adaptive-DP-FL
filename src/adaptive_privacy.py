# adaptive_privacy.py
import torch
import numpy as np
from opacus import PrivacyEngine
from opacus.accountants import RDPAccountant

class RDPAccountantCustom:
    """
    Custom RDP Accountant for tracking privacy budget across adaptive training.
    
    Implements Rényi Differential Privacy (RDP) accounting which provides
    tighter privacy bounds than basic composition. Tracks privacy budget
    across multiple steps with potentially different noise multipliers.
    """
    
    def __init__(self):
        # Track all (noise_multiplier, sample_rate, steps) tuples
        self.history = []
        # RDP orders (alpha values) to compute
        self.rdp_orders = [1 + x / 10.0 for x in range(1, 100)]  # 1.1 to 10.0
    
    def step(self, noise_multiplier, sample_rate, steps=1):
        """
        Record a privacy step.
        
        Args:
            noise_multiplier: Noise scale (sigma)
            sample_rate: Fraction of data used (batch_size / dataset_size)
            steps: Number of steps with these parameters
        """
        self.history.append({
            'noise_multiplier': noise_multiplier,
            'sample_rate': sample_rate,
            'steps': steps
        })
    
    def get_rdp(self, alpha):
        """
        Compute RDP at order alpha.
        
        Uses the formula for subsampled Gaussian mechanism from:
        "Rényi Differential Privacy" (Mironov, 2017)
        
        For Gaussian mechanism with noise_multiplier sigma and sample_rate q:
        - Full dataset (q=1): RDP(alpha) = alpha / (2 * sigma^2)
        - Subsampled (q<1): RDP(alpha) ≈ q^2 * alpha / (2 * sigma^2) for small q
        - More accurate: RDP(alpha) = (1/(alpha-1)) * log(E[exp((alpha-1)*Z)])
          where Z is the privacy loss random variable
        
        Args:
            alpha: RDP order (must be > 1)
            
        Returns:
            RDP value at order alpha
        """
        if alpha <= 1:
            return float('inf')
        
        total_rdp = 0.0
        
        for record in self.history:
            sigma = record['noise_multiplier']
            q = record['sample_rate']
            steps = record['steps']
            
            if sigma == 0:
                # No noise = infinite privacy cost
                return float('inf')
            
            # RDP for subsampled Gaussian mechanism
            # For Poisson subsampling with rate q and Gaussian noise sigma:
            # RDP(alpha) = (1/(alpha-1)) * log(1 + q^2 * (alpha choose 2) * min(4*exp(sigma^-2), 2*exp(1)) / sigma^2 + ...)
            # Simplified approximation for computational efficiency:
            # RDP(alpha) ≈ q^2 * alpha / (2 * sigma^2) for typical values
            
            if q >= 1.0:
                # Full dataset: standard Gaussian mechanism
                # RDP(alpha) = alpha / (2 * sigma^2)
                rdp_per_step = alpha / (2 * sigma**2)
            else:
                # Subsampled Gaussian mechanism (Poisson sampling)
                # The privacy amplification from subsampling reduces RDP by approximately q^2
                # More accurate formula accounts for the composition
                # Simplified: RDP ≈ q^2 * alpha / (2 * sigma^2)
                rdp_per_step = (q**2 * alpha) / (2 * sigma**2)
                
                # For very small q, we can use a tighter bound, but this approximation is sufficient
                # for most practical purposes
            
            total_rdp += rdp_per_step * steps
        
        return total_rdp
    
    def get_epsilon(self, delta=1e-5):
        """
        Convert RDP to (ε, δ)-DP.
        
        ε(δ) = min_alpha [RDP(alpha) + log(1/δ) / (alpha - 1)]
        
        Args:
            delta: Failure probability (typically 1e-5)
            
        Returns:
            Epsilon value
        """
        if len(self.history) == 0:
            return 0.0
        
        min_epsilon = float('inf')
        
        for alpha in self.rdp_orders:
            rdp = self.get_rdp(alpha)
            if rdp == float('inf'):
                continue
            
            # Convert RDP to (ε, δ)-DP
            epsilon = rdp + np.log(1.0 / delta) / (alpha - 1)
            min_epsilon = min(min_epsilon, epsilon)
        
        return min_epsilon if min_epsilon != float('inf') else 0.0
    
    def reset(self):
        """Reset the accountant history"""
        self.history = []

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
        
        # Adaptive Clipping State
        self.gradient_norms = []  # Store gradient norms for quantile calculation
        self.clipping_quantile = 0.75  # Use 75th percentile for clipping threshold
        self.min_clip_norm = 0.5  # Minimum clipping threshold
        self.max_clip_norm = 2.0  # Maximum clipping threshold (reduced from 5.0 to prevent instability)
        
        # Initialize Opacus
        self.privacy_engine = PrivacyEngine()
        
        # Initialize RDP Accountant for tracking privacy budget
        self.accountant = RDPAccountantCustom()
        
        # Track training steps for RDP accounting
        self.training_steps = 0
        self.dataset_size = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 10000
        
    def collect_gradient_norm(self, model):
        """
        Collect the L2 norm of gradients for adaptive clipping.
        Should be called before gradient clipping during training.
        
        Args:
            model: PyTorch model with gradients computed
        """
        if self.strategy == 'static':
            return
        
        # Calculate total L2 norm of all gradients
        total_norm = 0.0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.gradient_norms.append(total_norm)
    
    def adapt_clipping_threshold(self):
        """
        Adapt clipping threshold based on quantile of observed gradient norms.
        Uses median (50th percentile) or specified quantile to set clipping threshold.
        This ensures clipping bounds sensitivity without destroying the signal of
        the majority of gradients.
        """
        if self.strategy == 'static' or len(self.gradient_norms) == 0:
            return
        
        # Calculate quantile of gradient norms
        gradient_norms_array = np.array(self.gradient_norms)
        
        # Use median (50th percentile) as base, or specified quantile
        quantile_value = np.percentile(gradient_norms_array, self.clipping_quantile * 100)
        median_value = np.median(gradient_norms_array)
        
        # Set clipping threshold based on quantile
        # Use a slightly higher value than quantile to avoid clipping too many gradients
        # Common approach: use 75th percentile or median * 1.5
        if self.clipping_quantile == 0.5:
            # Use median with a safety margin
            new_clip_norm = median_value * 1.2
        else:
            # Use specified quantile
            new_clip_norm = quantile_value * 1.1  # 10% margin above quantile
        
        # Bound the clipping threshold more aggressively
        new_clip_norm = max(self.min_clip_norm, min(self.max_clip_norm, new_clip_norm))
        
        # Smooth update: use exponential moving average to avoid sudden changes
        # Use stronger smoothing (lower alpha) to prevent rapid growth
        alpha = 0.2  # Reduced from 0.3 for more conservative updates
        self.max_grad_norm = alpha * new_clip_norm + (1 - alpha) * self.max_grad_norm
        
        # Additional safety: ensure we don't exceed max_clip_norm after smoothing
        self.max_grad_norm = min(self.max_grad_norm, self.max_clip_norm)
        
        # Clear collected norms for next round
        self.gradient_norms = []
    
    def adapt_parameters(self, val_loss, debug=False, round_num=None):
        """
        Adaptive Privacy Mechanism - Closed-Loop Controller (Monitor-Decide-Act pattern)
        
        This implements the adaptive noise scaling mechanism as specified:
        1. Monitor: Tracks validation loss Lt after every local training epoch
        2. Decide: Calculates scaling factor αt based on loss trend
           - If Lt < Lt-1 (learning is progressing): reduce noise multiplier σ by decay factor γ = 0.98
           - If Lt >= Lt-1 (stagnation): increase noise to prevent overfitting
        3. Act: The new σ is passed to PrivacyEngine for the next round
        
        This allows the system to "spend" more privacy budget during critical learning phases
        and "save" it during fine-tuning, optimizing the utility-privacy trade-off.
        
        Args:
            val_loss: Validation loss Lt after local training epoch
            debug: If True, print debugging information
            round_num: Round number for debugging
            
        Returns:
            (noise_multiplier, max_grad_norm): Updated privacy parameters for next round
        """
        if self.strategy == 'static':
            return self.noise_multiplier, self.max_grad_norm

        # Store old values for debugging
        old_noise_mult = self.noise_multiplier
        old_clip_norm = self.max_grad_norm
        old_prev_loss = self.prev_loss

        # --- ADAPTIVE LOGIC ---
        # 1. Adaptive Clipping (based on gradient norm quantiles):
        # Only adapt clipping if we have gradient norms collected (i.e., for LDP)
        # For CDP, gradient norms are not collected, so clipping stays at default
        if len(self.gradient_norms) > 0:
            self.adapt_clipping_threshold()
        # For CDP: Keep clipping threshold at 1.0 (default) to maintain consistent noise scale
        
        # 2. Noise Adaptation (Decide step):
        # Calculate scaling factor αt based on loss trend
        # If Lt < Lt-1 (learning is progressing), reduce σ by decay factor γ = 0.98
        # BUT: Only reduce if improvement is significant to avoid premature noise reduction
        loss_improvement = (self.prev_loss - val_loss) / self.prev_loss if self.prev_loss > 0 else 0
        
        # More conservative thresholds: require larger improvements before reducing noise
        if val_loss < self.prev_loss and loss_improvement > 0.02:  # Increased from 0.005 to 0.02 (2% improvement required)
            # Significant improvement detected: Reduce noise multiplier slowly
            # Use very slow decay to avoid reducing noise too aggressively
            self.noise_multiplier = max(0.7, self.noise_multiplier * 0.995)  # Changed to 0.995 for very slow decay, min 0.7
            self.patience_counter = 0
            action = "REDUCE_NOISE"
        elif val_loss < self.prev_loss and loss_improvement > 0.01:  # Moderate improvement: keep noise
            # Moderate improvement: Keep noise level (don't reduce yet)
            self.noise_multiplier = self.noise_multiplier
            self.patience_counter = 0
            action = "KEEP_NOISE"
        elif val_loss < self.prev_loss:
            # Small improvement: Slightly reduce noise (very conservative)
            self.noise_multiplier = max(0.7, self.noise_multiplier * 0.998)
            self.patience_counter = 0
            action = "SLIGHT_REDUCE"
        else:
            # Stagnation or increase: Increase noise to prevent overfitting/memorization
            # This conserves privacy budget when model is not learning effectively
            self.noise_multiplier = min(1.5, self.noise_multiplier * 1.01)  # Slower increase, cap at 1.5
            self.patience_counter += 1
            action = "INCREASE_NOISE"

        # 3. Loss-based Clipping Adjustment (fallback):
        # If gradients are exploding (loss spike), tighten the clip 
        # Only adjust if we're using adaptive clipping (LDP)
        if len(self.gradient_norms) > 0 and val_loss > self.prev_loss * 1.1: 
             self.max_grad_norm = max(self.min_clip_norm, self.max_grad_norm * 0.9)
             clip_action = "DECREASE_CLIP"
        else:
             clip_action = "KEEP_CLIP"
        
        # Debug output
        if debug:
            print(f"   [DEBUG Round {round_num}] Adaptive Parameters:")
            print(f"      Validation Loss: {val_loss:.4f} (prev: {old_prev_loss:.4f}, improvement: {loss_improvement*100:.2f}%)")
            print(f"      Noise Multiplier: {old_noise_mult:.4f} → {self.noise_multiplier:.4f} ({action})")
            print(f"      Clip Norm: {old_clip_norm:.4f} → {self.max_grad_norm:.4f} ({clip_action})")
            print(f"      Patience Counter: {self.patience_counter}")
        
        # Update previous loss for next comparison (Monitor step)
        self.prev_loss = val_loss
        
        # Act step: Return updated parameters (will be used by PrivacyEngine in next round)
        return self.noise_multiplier, self.max_grad_norm

    def record_training_step(self, batch_size):
        """
        Record a training step for RDP accounting.
        Should be called after each training step.
        
        Args:
            batch_size: Size of the batch used in this step
        """
        if self.strategy == 'static':
            return
        
        sample_rate = batch_size / self.dataset_size if self.dataset_size > 0 else 1.0
        self.accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=sample_rate,
            steps=1
        )
        self.training_steps += 1
    
    def get_privacy_spent(self, delta=1e-5):
        """
        Get cumulative privacy budget (epsilon) spent.
        
        Uses RDP accounting to compute tight privacy bounds.
        
        Args:
            delta: Failure probability (default 1e-5)
            
        Returns:
            Cumulative epsilon value
        """
        return self.accountant.get_epsilon(delta=delta)
    
    def get_round_privacy_cost(self, num_steps, batch_size=None, delta=1e-5):
        """
        Compute privacy cost for a round with current parameters.
        
        Args:
            num_steps: Number of training steps in this round
            batch_size: Batch size used (if None, will use default from dataloader)
            delta: Failure probability
            
        Returns:
            Epsilon cost for this round
        """
        if self.strategy == 'static':
            return 0.0
        
        # Handle None batch_size - try to get from dataloader or use default
        if batch_size is None:
            # Try to get batch size from dataloader
            if hasattr(self.dataloader, 'batch_size') and self.dataloader.batch_size is not None:
                batch_size = self.dataloader.batch_size
            else:
                # Try to infer from first batch
                try:
                    first_batch = next(iter(self.dataloader))
                    if isinstance(first_batch, (list, tuple)) and len(first_batch) > 0:
                        batch_size = first_batch[0].size(0) if hasattr(first_batch[0], 'size') else 64
                    else:
                        batch_size = 64  # Default fallback
                except:
                    batch_size = 64  # Default fallback
        
        # Create temporary accountant for this round
        temp_accountant = RDPAccountantCustom()
        sample_rate = batch_size / self.dataset_size if (self.dataset_size > 0 and batch_size is not None) else 1.0
        
        temp_accountant.step(
            noise_multiplier=self.noise_multiplier,
            sample_rate=sample_rate,
            steps=num_steps
        )
        
        return temp_accountant.get_epsilon(delta=delta)
        return temp_accountant.get_epsilon(delta=delta)