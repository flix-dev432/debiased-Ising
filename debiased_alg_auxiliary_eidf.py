import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import copy
import gc
from scipy.optimize import minimize
import time
import sys
import wandb

# Check for device - use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def get_current_device():
    """Get the current device, handling multi-GPU scenarios"""
    print(f"[DEBUG] get_current_device: Checking CUDA availability")
    sys.stdout.flush()
    
    if torch.cuda.is_available():
        current_dev = torch.cuda.current_device()
        print(f"[DEBUG] get_current_device: CUDA available, current device: {current_dev}")
        sys.stdout.flush()
        return torch.device(f"cuda:{current_dev}")
    else:
        print(f"[DEBUG] get_current_device: CUDA not available, using CPU")
        sys.stdout.flush()
        return torch.device("cpu")

def ensure_device_placement(tensor_or_model, device=None):
    """Ensure tensor or model is on the correct device"""
    if device is None:
        device = get_current_device()
    
    if hasattr(tensor_or_model, 'to'):
        return tensor_or_model.to(device)
    return tensor_or_model

def gpu_memory_cleanup():
    """Clean up GPU memory to prevent hanging - conservative approach"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Free unused cached memory
        gc.collect()              # Clean up Python objects
        # torch.cuda.synchronize()

def safe_gpu_cleanup():
    """Safe GPU cleanup that doesn't interfere with other workers"""
    if torch.cuda.is_available():
        # Only synchronize the current stream - does not affect other workers
        torch.cuda.synchronize()
        # Do NOT call empty_cache() here to avoid interfering with other workers

def safe_to_device(data, device=None):
    """Safely move data to device with error handling"""
    if device is None:
        device = get_current_device()
    
    print(f"[DEBUG] safe_to_device: Moving data to {device}")
    sys.stdout.flush()
    
    try:
        if hasattr(data, 'to'):
            result = data.to(device)
            print(f"[DEBUG] safe_to_device: Successfully moved tensor to {device}")
            sys.stdout.flush()
            return result
        elif isinstance(data, (list, tuple)):
            return [safe_to_device(item, device) for item in data]
        else:
            return data
    except Exception as e:
        print(f"Warning: Failed to move data to device {device}: {e}")
        return data

def setup_worker_gpu_context():
    """Set up proper GPU context for multi-worker scenarios"""
    if torch.cuda.is_available():
        # Initialize CUDA context for this worker (safe for multiple workers)
        torch.cuda.init()
        # Only synchronize - do NOT clear cache to avoid interfering with other workers
        torch.cuda.synchronize()
        
        current_device = torch.cuda.current_device()
        print(f"Worker initialized on GPU {current_device}")
        return torch.device(f"cuda:{current_device}")
    else:
        return torch.device("cpu")

def sigmoid(x):
    """Sigmoid function for probability calculation"""
    return 1 / (1 + np.exp(-x))

# Factory functions for model types
def get_theta_model(model_type, input_dim):
    """Get theta model based on model type"""
    current_device = get_current_device()
    if model_type == 'linear':
        return LinearTheta(input_dim).to(current_device)
    elif model_type == 'nn':
        return ThetaNet(input_dim).to(current_device)
    elif model_type == 'constant':
        return ConstantTheta().to(current_device)
    elif model_type == 'X_only':
        return MainTermsTheta(input_dim).to(current_device)
    else:
        raise ValueError(f"Unknown theta model type: {model_type}. Choose from 'linear', 'nn', 'constant', or 'X_only'.")

def get_alpha_model(model_type, input_dim, alpha_nn_setup=None):
    """Get alpha model based on model type
    
    Parameters:
    -----------
    model_type : str
        Type of alpha model ('linear', 'nn', 'polynomial')
    input_dim : int  
        Input dimension
    alpha_nn_setup : list, optional
        Hidden layer architecture for neural network alpha model.
        Only used when model_type='nn'. Default: [100, 50]
        Example: [64, 64] creates two hidden layers with 64 nodes each
    """
    current_device = get_current_device()
    if model_type == 'linear':
        return LinearAlpha(input_dim, current_device)
    elif model_type == 'nn':
        return AlphaNet(input_dim, current_device, alpha_nn_setup)
    elif model_type == 'polynomial':
        return PolynomialAlpha(input_dim, current_device)
    else:
        raise ValueError(f"Unknown alpha model type: {model_type}. Choose from 'linear', 'nn', or 'polynomial'.")

# Linear Model for theta
class LinearTheta(nn.Module):
    def __init__(self, input_dim):
        super(LinearTheta, self).__init__()
        # Simple linear layer for b0 + b1*A + b2*X1 + b3*X2 + ...
        self.linear = nn.Linear(input_dim + 1, 1)  # +1 for treatment A
    
    def forward(self, a, x):
        # Concatenate treatment A and covariates X
        inputs = torch.cat([a.unsqueeze(1), x], dim=1)
        return self.linear(inputs)

# Constant Model for theta (theta = b0, constant across all inputs)
class ConstantTheta(nn.Module):
    def __init__(self):
        super(ConstantTheta, self).__init__()
        # Just a single parameter b0
        self.bias = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, a, x):
        # Return the same constant value for all inputs
        batch_size = a.shape[0]
        return self.bias.expand(batch_size)

# Main Terms Linear Model for theta (theta = b0 + b1*X1 + b2*X2 + ..., without A)
class MainTermsTheta(nn.Module):
    def __init__(self, input_dim):
        super(MainTermsTheta, self).__init__()
        # Linear layer for b0 + b1*X1 + b2*X2 + ... (only X variables, no treatment A)
        self.linear = nn.Linear(input_dim, 1)  # input_dim for X variables only
    
    def forward(self, a, x):
        # Only use covariates X, ignore treatment A
        return self.linear(x)

# Linear Model for alpha components
class LinearAlphaComponent(nn.Module):
    def __init__(self, input_dim):
        super(LinearAlphaComponent, self).__init__()
        # Simple linear layer for b0 + b1*X1 + b2*X2 + ...
        self.linear = nn.Linear(input_dim, 1)
    
    def forward(self, x):
        return self.linear(x)

# Alpha Model combining the two components
class LinearAlpha:
    def __init__(self, input_dim, device=None):
        if device is None:
            device = get_current_device()
        self.device = device
        self.g0 = LinearAlphaComponent(input_dim).to(device)  # For a=0
        self.g1 = LinearAlphaComponent(input_dim).to(device)  # For a=1
    
    def __call__(self, a, x):
        """
        Compute alpha(a, x) = a*g1(x) + (1-a)*g0(x)
        """
        a = ensure_device_placement(a, self.device)
        x = ensure_device_placement(x, self.device)
        g0_output = self.g0(x)
        g1_output = self.g1(x)
        
        # Use broadcasting for element-wise calculation
        return a.unsqueeze(1) * g1_output + (1 - a.unsqueeze(1)) * g0_output
    
    def parameters(self):
        """Return all parameters for optimization"""
        return list(self.g0.parameters()) + list(self.g1.parameters())

# Neural Network for theta
class ThetaNet(nn.Module):
    def __init__(self, input_dim):
        super(ThetaNet, self).__init__()
        
        # Hidden layers
        self.layers = nn.Sequential(
            nn.Linear(input_dim + 1, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
    
    def forward(self, a, x):
        # Concatenate treatment A and covariates X
        inputs = torch.cat([a.unsqueeze(1), x], dim=1)
        return self.layers(inputs)

# Neural Network for alpha components
class AlphaComponentNet(nn.Module):
    def __init__(self, input_dim, hidden_layers=None):
        super(AlphaComponentNet, self).__init__()
        
        # Default architecture for backward compatibility
        if hidden_layers is None:
            hidden_layers = [100, 50]
        
        # Build the layers dynamically
        layers = []
        prev_dim = input_dim
        
        # Add hidden layers
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            # layers.append(nn.Dropout(p=0.05))  # Add dropout with p=0.05
            prev_dim = hidden_dim
        
        # Add final output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)

# Alpha Model combining the two components
class AlphaNet:
    def __init__(self, input_dim, device=None, hidden_layers=None):
        if device is None:
            device = get_current_device()
        self.device = device
        self.g0 = AlphaComponentNet(input_dim, hidden_layers).to(device)  # For a=0
        self.g1 = AlphaComponentNet(input_dim, hidden_layers).to(device)  # For a=1
    
    def __call__(self, a, x):
        """
        Compute alpha(a, x) = a*g1(x) + (1-a)*g0(x)
        """
        a = ensure_device_placement(a, self.device)
        x = ensure_device_placement(x, self.device)
        g0_output = self.g0(x)
        g1_output = self.g1(x)
        
        # Use broadcasting for element-wise calculation
        return a.unsqueeze(1) * g1_output + (1 - a.unsqueeze(1)) * g0_output
    
    def parameters(self):
        """Return all parameters for optimization"""
        return list(self.g0.parameters()) + list(self.g1.parameters())

# Polynomial Alpha Component with all two-way interactions
class PolynomialAlphaComponent(nn.Module):
    def __init__(self, input_dim):
        super(PolynomialAlphaComponent, self).__init__()
        
        self.input_dim = input_dim
        
        # Calculate the number of features:
        # input_dim main terms + C(input_dim, 2) two-way interactions
        n_interactions = input_dim * (input_dim - 1) // 2
        self.total_features = input_dim + n_interactions

        # Linear layer for polynomial features
        self.linear = nn.Linear(self.total_features, 1, bias=True)
        
        # Initialize with smaller weights due to high dimensionality
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        nn.init.constant_(self.linear.bias, 0)
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Create polynomial features efficiently using Kronecker-like operations
        # This vectorized approach replaces for-loop iteration for better GPU performance
        
        # Main terms (original features)
        main_terms = x  # Shape: (batch_size, input_dim)
        
        # Two-way interactions using vectorized operations
        # Create all pairwise products using outer product approach
        x_expanded_i = x.unsqueeze(2)  # Shape: (batch_size, input_dim, 1)
        x_expanded_j = x.unsqueeze(1)  # Shape: (batch_size, 1, input_dim)
        
        # Compute all pairwise products
        all_products = x_expanded_i * x_expanded_j  # Shape: (batch_size, input_dim, input_dim)
        
        # Extract upper triangular part (excluding diagonal) for unique interactions
        # Create mask for upper triangular part (i < j)
        triu_mask = torch.triu(torch.ones(self.input_dim, self.input_dim, device=x.device), diagonal=1).bool()
        
        # Extract interactions using the mask
        interaction_terms = all_products[:, triu_mask]  # Shape: (batch_size, n_interactions)
        
        # Concatenate main terms and interaction terms
        poly_x = torch.cat([main_terms, interaction_terms], dim=1)
        
        return self.linear(poly_x)

# Polynomial Alpha Model combining the two components with elastic net
class PolynomialAlpha:
    def __init__(self, input_dim, device=None):
        if device is None:
            device = get_current_device()
        self.device = device
        self.input_dim = input_dim
        self.g0 = PolynomialAlphaComponent(input_dim).to(device)  # For a=0
        self.g1 = PolynomialAlphaComponent(input_dim).to(device)  # For a=1
        
        # Store total number of features for regularization
        n_interactions = input_dim * (input_dim - 1) // 2
        self.total_features = (input_dim + n_interactions) * 2
        
        print(f"    Polynomial alpha model: {input_dim} main terms + {n_interactions} interaction terms = {self.total_features} total features")
    
    def __call__(self, a, x):
        """
        Compute alpha(a, x) = a*g1(x) + (1-a)*g0(x)
        """
        a = ensure_device_placement(a, self.device)
        x = ensure_device_placement(x, self.device)
        g0_output = self.g0(x)
        g1_output = self.g1(x)
        
        # Use broadcasting for element-wise calculation
        return a.unsqueeze(1) * g1_output + (1 - a.unsqueeze(1)) * g0_output
    
    def parameters(self):
        """Return all parameters for optimization"""
        return list(self.g0.parameters()) + list(self.g1.parameters())
  
def get_alpha_regularization(alpha_model, lambda_alpha, l1_ratio=0.5, exclude_bias=False):
    """
    Regularisation for alpha model.
      - Linear: L2 squared only
      - Polynomial: elastic net = l1_ratio * L1 + (1 - l1_ratio) * L2_squared
      - NN: L1 + L2 squared regularisation
    If exclude_bias=True, biases are skipped in both L1 and L2 terms.
    """
    # Collect (name, param) from g0 and g1 (works for all your alpha wrappers)
    if hasattr(alpha_model, 'g0') and hasattr(alpha_model, 'g1'):
        named_params = []
        for prefix, mod in (('g0', alpha_model.g0), ('g1', alpha_model.g1)):
            named_params.extend((f"{prefix}.{n}", p) for n, p in mod.named_parameters())
    else:
        # Fallback if alpha_model itself is an nn.Module (not the case here, but safe)
        named_params = list(alpha_model.named_parameters())

    # Decide what type of alpha model this is
    is_polynomial = isinstance(getattr(alpha_model, 'g0', None), PolynomialAlphaComponent) and isinstance(getattr(alpha_model, 'g1', None), PolynomialAlphaComponent)
    is_nn = isinstance(getattr(alpha_model, 'g0', None), AlphaComponentNet) and isinstance(getattr(alpha_model, 'g1', None), AlphaComponentNet)

    l2_sq = 0.0
    l1 = 0.0
    for name, p in named_params:
        if not p.requires_grad:
            continue
        if exclude_bias and name.rsplit('.', 1)[-1] == 'bias':
            continue

        # squared L2 always accumulated
        l2_sq = l2_sq + p.pow(2).sum()

        # L1 for polynomial (elastic net) or NN (L1 + L2)
        if is_polynomial or is_nn:
            l1 = l1 + p.abs().sum()

    if is_polynomial:
        penalty = l1_ratio * l1 + (1.0 - l1_ratio) * l2_sq
    elif is_nn:
        penalty = l1 + l2_sq  # L1 + L2 squared for NN
    else:
        penalty = l2_sq  # L2 squared only for linear

    return lambda_alpha * penalty

# Train theta model
def train_theta(theta_model, a, x, y, lambda_theta=0.001, epochs=200, batch_size=128, lr=0.01, 
               fold_idx=None, run_id=None):
    """Train the theta model (works for both linear and NN)"""
    # Ensure all inputs are on the same device as the model
    device = next(theta_model.parameters()).device
    a = safe_to_device(a, device)
    x = safe_to_device(x, device)
    y = safe_to_device(y, device)
    
    # Clean up GPU memory before training
    gpu_memory_cleanup()
    
    # Handle squared L2 by optimiser (exclude bias) 
    decay, no_decay = [], []
    for name, param in theta_model.named_parameters():
        if not param.requires_grad:
            continue
        if name.rsplit('.', 1)[-1] == 'bias':   # Exclude bias from weight decay # name.endswith('bias')
            no_decay.append(param)
        else:
            decay.append(param)

    optimizer = optim.AdamW(
        [
            {"params": decay, "weight_decay": lambda_theta},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )
    
    # Create dataset and dataloader
    dataset = TensorDataset(a, x, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Loss function: numerically stable BCE with logits (mean over batch)
    bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')
    
    # Initialize learning curve tracking
    theta_losses = []
    
    # Training loop
    theta_model.train()
    
    for epoch in range(epochs):
        epoch_total_loss = 0
        epoch_total_samples = 0
        
        for batch_a, batch_x, batch_y in dataloader:
            optimizer.zero_grad(set_to_none=True)
            
            # Get theta predictions - vectorized processing
            logits = theta_model(batch_a, batch_x).squeeze(-1)  # shape safety
            targets = batch_y.float().view_as(logits)           # shape safety
            
            # Negative log-likelihood (Binary cross-entropy) 
            # Unregularised data loss (mean over batch, numerically stable)
            bce = bce_loss(logits, targets)

            # Total loss (regularisation via AdamW's weight_decay)
            loss = bce
            
            loss.backward()
            optimizer.step()
            
            # Accumulate unregularised loss for correct per-sample epoch average
            bs = targets.size(0)
            epoch_total_loss += bce.item() * bs
            epoch_total_samples += bs
        
        # Average loss per sample over the epoch (unregularised)
        avg_epoch_loss = epoch_total_loss / max(1, epoch_total_samples)
        theta_losses.append(avg_epoch_loss)
    
    # Return model and learning curve
    return theta_model, theta_losses

def plot_learning_curves_wandb(theta_curves, alpha_curves, run_id, pair_info, model_type, method):
    """
    Plot learning curves using Weights & Biases
    
    Parameters:
    -----------
    theta_curves : list
        List of learning curves for theta models (one per fold)
    alpha_curves : list
        List of learning curves for alpha models (one per fold)
    run_id : str
        Unique identifier for this run
    pair_info : dict
        Information about the current pair being processed
    model_type : str
        Type of model being used
    method : str
        Method being used (autoDML or autoTML)
    """
    try:
        import matplotlib.pyplot as plt
        
        # Create theta learning curve plot
        fig_theta, ax_theta = plt.subplots(figsize=(10, 6))
        for fold_idx, losses in enumerate(theta_curves):
            epochs = range(1, len(losses) + 1)
            ax_theta.plot(epochs, losses, label=f'Fold {fold_idx + 1}', alpha=0.7, linewidth=2)
        
        ax_theta.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax_theta.set_ylabel('Average Loss', fontsize=14, fontweight='bold')
        ax_theta.set_title(f'Theta Training - {method} - Pair ({pair_info["y"]}, {pair_info["a"]}) - {model_type}', 
                          fontsize=16, fontweight='bold')
        ax_theta.legend(fontsize=12)
        ax_theta.grid(True, alpha=0.3)
        
        # Create alpha learning curve plot
        fig_alpha, ax_alpha = plt.subplots(figsize=(10, 6))
        for fold_idx, losses in enumerate(alpha_curves):
            epochs = range(1, len(losses) + 1)
            ax_alpha.plot(epochs, losses, label=f'Fold {fold_idx + 1}', alpha=0.7, linewidth=2)
        
        ax_alpha.set_xlabel('Epochs', fontsize=14, fontweight='bold')
        ax_alpha.set_ylabel('Average Loss', fontsize=14, fontweight='bold')
        ax_alpha.set_title(f'Alpha Training - {method} - Pair ({pair_info["y"]}, {pair_info["a"]}) - {model_type}', 
                          fontsize=16, fontweight='bold')
        ax_alpha.legend(fontsize=12)
        ax_alpha.grid(True, alpha=0.3)
        
        # Log to wandb
        wandb.log({
            f"theta_learning_curve_pair_{pair_info['y']}_{pair_info['a']}": wandb.Image(fig_theta),
            f"alpha_learning_curve_pair_{pair_info['y']}_{pair_info['a']}": wandb.Image(fig_alpha)
        })
        
        # Close figures to save memory
        plt.close(fig_theta)
        plt.close(fig_alpha)
        
    except Exception as e:
        print(f"Warning: Could not create wandb plots: {e}")

# Implementation of autoDML algorithm 
def autoDML(X, A, Y, model_type_theta='linear', model_type_alpha='linear', n_splits=5, lambda_theta=0.001, lambda_alpha=0.001, l1_ratio=0.5,
            theta_epochs=200, alpha_epochs=300, batch_size_theta=128, batch_size_alpha=None, lr_theta=0.01, lr_alpha=0.01, 
            stabilization=True, seed=None, alpha_nn_setup=None, run_id=None, pair_info=None):
    """
    Implement the autoDML algorithm with cross-fitting using specified model types
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary covariates (n_samples x p)
    A : numpy.ndarray
        Binary treatment assignment
    Y : numpy.ndarray
        Binary outcome
    model_type_theta : str
        Type of model to use for theta: 'linear', 'nn' (neural network), 'constant' (theta as constant), or 'X_only' (theta as main terms linear model of X only)
    model_type_alpha : str
        Type of model to use for alpha: 'linear', 'nn' (neural network), or 'polynomial' (polynomial with two-way interactions)
    n_splits : int
        Number of cross-fitting splits
    lambda_theta, lambda_alpha : float
        Regularization parameters
    theta_epochs, alpha_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr_theta, lr_alpha : float
        Learning rates
    stabilization : bool
        Whether to apply TMLE-inspired stabilization
    seed : int, optional
        Random seed for reproducibility
    conv_threshold : float
        Convergence threshold for early stopping
    patience : int
        Number of epochs with small changes before early stopping
    run_id : str, optional
        Unique identifier for this run (for wandb tracking)
    pair_info : dict, optional
        Information about the current pair being processed
        
    Returns:
    --------
    dict
        Dictionary with estimation results
    """
    start_time = time.time()
    
    # Use the current device context set by the worker
    # Do NOT reinitialize GPU context here to avoid conflicts
    current_device = get_current_device()

    # Handle backward compatibility for batch sizes
    if batch_size_alpha is None:
        batch_size_alpha = batch_size_theta
    
    # Set seed if provided (for parallel runs)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    n = len(X)
    input_dim = X.shape[1]
    
    # Convert data to PyTorch tensors using current device context
    X_tensor = torch.FloatTensor(X.copy()).to(current_device)
    A_tensor = torch.FloatTensor(A.copy()).to(current_device)
    Y_tensor = torch.FloatTensor(Y.copy()).to(current_device)
    
    # Step 0: Split data into 85% training and 15% validation
    print("Step 0: Splitting data into training (85%) and validation (15%) sets...")
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    n_val = int(0.15 * n)
    n_train = n - n_val
    
    # Split indices
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    print(f"  Training set size: {n_train} samples")
    print(f"  Validation set size: {n_val} samples")
    
    # Extract training and validation data
    X_train_full = X_tensor[train_indices]
    A_train_full = A_tensor[train_indices]
    Y_train_full = Y_tensor[train_indices]
    
    X_val = X_tensor[val_indices]
    A_val = A_tensor[val_indices]
    Y_val = Y_tensor[val_indices]
    
    # Create data splits for cross-fitting on the training set only
    train_indices_shuffled = np.arange(n_train)
    np.random.shuffle(train_indices_shuffled)
    split_indices = np.array_split(train_indices_shuffled, n_splits)
    
    # Maps each training sample to its fold
    fold_mapping = np.zeros(n_train, dtype=int)
    for j, fold_indices in enumerate(split_indices):
        fold_mapping[fold_indices] = j
    
    # Initialize lists to store cross-fitted models and learning curves
    theta_models = []
    alpha_models = []
    theta_learning_curves = []
    alpha_learning_curves = []
    
    # Step 1: Cross-fit theta models
    print("Step 1: Cross-fitting theta models...")
    theta_training_start = time.time()  # Start timing theta training
    for s in range(n_splits):
        print(f"Training theta model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Training data for this split
        X_train = X_tensor[train_indices]
        A_train = A_tensor[train_indices]
        Y_train = Y_tensor[train_indices]
        
        # Create and train theta model
        theta_model = get_theta_model(model_type_theta, input_dim)
        theta_model, theta_losses = train_theta(theta_model, A_train, X_train, Y_train, 
                               lambda_theta=lambda_theta, epochs=theta_epochs, 
                               batch_size=batch_size_theta, lr=lr_theta,
                               fold_idx=s, run_id=run_id)
        
        # Store theta model and learning curve
        theta_models.append(theta_model)
        theta_learning_curves.append(theta_losses)
    
    theta_training_time = time.time() - theta_training_start  # Calculate theta training time
    # Step 2: Cross-fit alpha models using pre-trained theta models
    print("Step 2: Cross-fitting alpha models...")
    alpha_training_start = time.time()  # Start timing alpha training
    
    # PRE-COMPUTE: Calculate theta predictions and sigmoid derivatives for all training samples
    # This avoids repeated computation and device transfers during alpha training
    print("  Pre-computing theta predictions for alpha training...")
    theta_predictions = torch.zeros(n_train, device=current_device)
    sigmoid_derivatives = torch.zeros(n_train, device=current_device)
    
    # Pre-compute predictions using cross-fitted theta models
    for i in range(n_train):
        fold = fold_mapping[i]
        theta_model = theta_models[fold]
        theta_model.eval()
        
        with torch.no_grad():
            theta_pred = theta_model(A_train_full[i:i+1], X_train_full[i:i+1]).squeeze()
            sigmoid_theta = torch.sigmoid(theta_pred)
            sigmoid_deriv = sigmoid_theta * (1 - sigmoid_theta)
            
            theta_predictions[i] = theta_pred
            sigmoid_derivatives[i] = sigmoid_deriv
    
    for s in range(n_splits):
        print(f"Training alpha model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Initialize alpha model
        alpha_model = get_alpha_model(model_type_alpha, input_dim, alpha_nn_setup)
        optimizer = optim.Adam(alpha_model.parameters(), lr=lr_alpha)  
        
        # Create dataset for training with pre-computed theta info
        train_dataset = TensorDataset(
            torch.tensor(train_indices, dtype=torch.long, device=current_device),
            A_train_full[train_indices], 
            X_train_full[train_indices],
            theta_predictions[train_indices],
            sigmoid_derivatives[train_indices]
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_alpha, shuffle=True)
        
        # Training loop
        if hasattr(alpha_model, 'g0') and hasattr(alpha_model, 'g1'):
            alpha_model.g0.train()
            alpha_model.g1.train()
        
        # For learning curve tracking
        alpha_losses = []
        
        for epoch in range(alpha_epochs):
            epoch_total_loss = 0
            epoch_total_samples = 0
            
            for batch_indices, batch_a, batch_x, batch_theta_preds, batch_sigmoid_derivs in train_dataloader:
                optimizer.zero_grad(set_to_none=True)
                
                # Use pre-computed sigmoid derivatives (no theta model calls needed!)
                sigmoid_derivs = batch_sigmoid_derivs
                
                # Alpha for current (a,x) pairs
                alpha_axs = alpha_model(batch_a, batch_x).squeeze(-1)
                
                # Compute alpha(1,x) and alpha(0,x) for all x in this batch
                alpha_1xs = alpha_model.g1(batch_x).squeeze(-1)
                alpha_0xs = alpha_model.g0(batch_x).squeeze(-1)
                alpha_diffs = alpha_1xs - alpha_0xs
                
                # Loss terms (vectorized) - using pre-computed sigmoid derivatives
                first_terms = sigmoid_derivs * alpha_axs**2
                second_terms = -2 * alpha_diffs
                
                # Batch loss
                batch_losses = first_terms + second_terms
                batch_mean_loss = batch_losses.mean()
                
                # Add regularization (elastic net for polynomial, L1+L2 for NN, L2 for linear)
                reg_penalty = get_alpha_regularization(alpha_model, lambda_alpha, l1_ratio=l1_ratio, exclude_bias=True)
                loss = batch_mean_loss + reg_penalty
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate for correct averaging
                # Per-sample epoch average of the unregularised loss
                bs = batch_a.size(0) 
                epoch_total_loss += batch_mean_loss.item() * bs
                epoch_total_samples += bs
            
            # Calculate correct average loss per sample
            avg_epoch_loss = epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else epoch_total_loss
            alpha_losses.append(avg_epoch_loss)
        
        # Store alpha model and learning curve
        alpha_models.append(alpha_model)
        alpha_learning_curves.append(alpha_losses)
    
    alpha_training_time = time.time() - alpha_training_start  # Calculate alpha training time
    
    # Note: Learning curves will be plotted in the main process to ensure wandb compatibility
    
    # Step 2.5: Compute validation losses
    print("Step 2.5: Computing validation losses...")
    
    # Train an extra theta model using the entire training set
    print("  Training extra theta model on full training set...")
    extra_theta_model = get_theta_model(model_type_theta, input_dim)
    extra_theta_model, _ = train_theta(extra_theta_model, A_train_full, X_train_full, Y_train_full, 
                           lambda_theta=lambda_theta, epochs=theta_epochs, 
                           batch_size=batch_size_theta, lr=lr_theta,
                           fold_idx=-1, run_id=run_id)
    
    # Compute validation losses for each alpha model
    alpha_validation_losses = []
    
    for k in range(n_splits):
        print(f"  Computing validation loss for alpha model {k+1}/{n_splits}")
        alpha_model = alpha_models[k]
        
        # Compute theta predictions and sigmoid derivatives on validation set
        val_theta_predictions = torch.zeros(n_val, device=current_device)
        val_sigmoid_derivatives = torch.zeros(n_val, device=current_device)
        
        extra_theta_model.eval()
        with torch.no_grad():
            for i in range(n_val):
                theta_pred = extra_theta_model(A_val[i:i+1], X_val[i:i+1]).squeeze()
                sigmoid_theta = torch.sigmoid(theta_pred)
                sigmoid_deriv = sigmoid_theta * (1 - sigmoid_theta)
                
                val_theta_predictions[i] = theta_pred
                val_sigmoid_derivatives[i] = sigmoid_deriv
        
        # Compute alpha loss on validation set
        val_dataset = TensorDataset(A_val, X_val, val_theta_predictions, val_sigmoid_derivatives)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_alpha, shuffle=False)
        
        total_val_loss = 0
        total_val_samples = 0
        
        alpha_model.g0.eval()
        alpha_model.g1.eval()
        
        with torch.no_grad():
            for batch_a, batch_x, batch_theta_preds, batch_sigmoid_derivs in val_dataloader:
                # Alpha for current (a,x) pairs
                alpha_axs = alpha_model(batch_a, batch_x).squeeze(-1)
                
                # Compute alpha(1,x) and alpha(0,x) for all x in this batch
                alpha_1xs = alpha_model.g1(batch_x).squeeze(-1)
                alpha_0xs = alpha_model.g0(batch_x).squeeze(-1)
                alpha_diffs = alpha_1xs - alpha_0xs
                
                # Loss terms (vectorized) - using pre-computed sigmoid derivatives
                first_terms = batch_sigmoid_derivs * alpha_axs**2
                second_terms = -2 * alpha_diffs
                
                # Batch total loss (sum instead of mean, without regularization for validation)
                batch_losses = first_terms + second_terms
                batch_total_loss = batch_losses.sum()
                
                # Accumulate for correct averaging
                total_val_loss += batch_total_loss.item()
                total_val_samples += batch_a.size(0)
        
        # Calculate correct average validation loss per sample
        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else total_val_loss
        alpha_validation_losses.append(avg_val_loss)
        print(f"    Alpha model {k+1} validation loss: {avg_val_loss:.6f}")
    
    # Compute mean validation loss
    mean_alpha_validation_loss = np.mean(alpha_validation_losses)
    print(f"  Mean alpha validation loss: {mean_alpha_validation_loss:.6f}")
    
    # Step 3: Apply TMLE-inspired stabilization (optional)
    if stabilization:
        print("Step 3: Applying TMLE-inspired stabilization...")
        
        # Use larger batch size for evaluation to reduce loop iterations
        batch_size_stab = min(5000, n_train)  # Use larger batch size based on training set
        n_batches_stab = int(np.ceil(n_train / batch_size_stab))
        
        numerator = 0
        denominator = 0
        
        for batch in range(n_batches_stab):
            start_idx = batch * batch_size_stab
            end_idx = min((batch + 1) * batch_size_stab, n_train)
            batch_indices = np.arange(start_idx, end_idx)
            
            batch_num = 0
            batch_denom = 0
            
            # Process samples in batch
            for i in batch_indices:
                fold = fold_mapping[i]
                
                theta_model = theta_models[fold]
                alpha_model = alpha_models[fold]
                
                with torch.no_grad():
                    # Calculate alpha difference
                    alpha_1x = alpha_model.g1(X_train_full[i:i+1]).item()
                    alpha_0x = alpha_model.g0(X_train_full[i:i+1]).item()
                    alpha_diff = alpha_1x - alpha_0x
                    
                    # Calculate denominator term
                    theta_ax = theta_model(A_train_full[i:i+1], X_train_full[i:i+1]).item()
                    sigmoid_theta = sigmoid(theta_ax)
                    sigmoid_deriv = sigmoid_theta * (1 - sigmoid_theta)
                    alpha_ax = alpha_model(A_train_full[i:i+1], X_train_full[i:i+1]).item()
                    denom_term = sigmoid_deriv * (alpha_ax ** 2)
                    
                    batch_num += alpha_diff
                    batch_denom += denom_term
            
            numerator += batch_num
            denominator += batch_denom
        
        # Calculate epsilon_n
        epsilon_n = numerator / denominator if denominator != 0 else 1.0
        print(f"  Stabilization factor epsilon_n = {epsilon_n:.4f}")
        
        # Store the epsilon_n value for scaling the alpha outputs
        stabilization_factor = epsilon_n
    else:
        stabilization_factor = 1.0  # No scaling if stabilization is disabled
    
    # Step 4: Compute one-step estimator
    print("Step 4: Computing final estimate...")
    
    # Use larger batch size for evaluation
    batch_size_eval = min(5000, n_train)  # Use larger batch size based on training set
    n_batches = int(np.ceil(n_train / batch_size_eval))
    
    sum_term = 0
    
    for batch in range(n_batches):
        start_idx = batch * batch_size_eval
        end_idx = min((batch + 1) * batch_size_eval, n_train)
        batch_indices = np.arange(start_idx, end_idx)
        
        batch_sum = 0
        
        # Process samples in batch
        for i in batch_indices:
            fold = fold_mapping[i]
            
            # Get models trained without this sample
            theta_model = theta_models[fold]
            alpha_model = alpha_models[fold]
            
            # Get data for this sample
            x_i = X_train_full[i:i+1]
            a_i = A_train_full[i:i+1]
            y_i = Y_train_full[i:i+1].item()
            
            with torch.no_grad():
                # First term: theta(1,X) - theta(0,X)
                theta_1x = theta_model(torch.ones_like(a_i), x_i).item()
                theta_0x = theta_model(torch.zeros_like(a_i), x_i).item()
                theta_diff = theta_1x - theta_0x
                
                # Second term: (sigma(theta(A,X)) - Y) * alpha(A,X) * epsilon_n
                theta_ax = theta_model(a_i, x_i).item()
                sigma_theta = sigmoid(theta_ax)
                alpha_ax = alpha_model(a_i, x_i).item() * stabilization_factor  # Scale alpha output
                second_term = (sigma_theta - y_i) * alpha_ax
                
                # Combined term for this sample
                term_i = theta_diff - second_term
                batch_sum += term_i
        
        sum_term += batch_sum
    
    # Final estimate
    psi_hat = sum_term / n_train
    
    # Step 5: Compute standard error
    print("Step 5: Computing standard error...")
    
    sum_squared_diff = 0
    
    for batch in range(n_batches):
        start_idx = batch * batch_size_eval
        end_idx = min((batch + 1) * batch_size_eval, n_train)
        batch_indices = np.arange(start_idx, end_idx)
        
        batch_sq_diff = 0
        
        for i in batch_indices:
            # Get the model fold for this sample
            fold = fold_mapping[i]
            
            # Get models trained without this sample
            theta_model = theta_models[fold]
            alpha_model = alpha_models[fold]
            
            # Get data for this sample
            x_i = X_train_full[i:i+1]
            a_i = A_train_full[i:i+1]
            y_i = Y_train_full[i:i+1].item()
            
            with torch.no_grad():
                # First term: theta(1,X) - theta(0,X)
                theta_1x = theta_model(torch.ones_like(a_i), x_i).item()
                theta_0x = theta_model(torch.zeros_like(a_i), x_i).item()
                theta_diff = theta_1x - theta_0x
                
                # Second term: (sigma(theta(A,X)) - Y) * alpha(A,X) * epsilon_n
                theta_ax = theta_model(a_i, x_i).item()
                sigma_theta = sigmoid(theta_ax)
                alpha_ax = alpha_model(a_i, x_i).item() * stabilization_factor  # Scale alpha output
                second_term = (sigma_theta - y_i) * alpha_ax
                
                # Combined term for this sample
                term_i = theta_diff - second_term
                
                # Squared difference for variance estimation
                squared_diff = (term_i - psi_hat) ** 2
                batch_sq_diff += squared_diff
        
        sum_squared_diff += batch_sq_diff
    
    # Estimated variance
    V_hat = sum_squared_diff / n_train
    
    # Standard error
    se_hat = np.sqrt(V_hat / n_train)
    
    # 95% confidence interval
    ci_lower = psi_hat - 1.96 * se_hat
    ci_upper = psi_hat + 1.96 * se_hat
    
    # Total execution time
    execution_time = time.time() - start_time
    
    # Return results
    results = {
        'psi_hat': psi_hat,
        'se_hat': se_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'variance': V_hat,
        'model_type_theta': model_type_theta,
        'model_type_alpha': model_type_alpha,
        'stabilization': stabilization,
        'stabilization_factor': stabilization_factor if stabilization else None,
        'execution_time': execution_time,
        'theta_training_time': theta_training_time,
        'alpha_training_time': alpha_training_time,
        'theta_learning_curves': theta_learning_curves,
        'alpha_learning_curves': alpha_learning_curves,
        'alpha_validation_losses': alpha_validation_losses,
        'mean_alpha_validation_loss': mean_alpha_validation_loss
    }
    
    return results

# Implementation of autoTML algorithm
def autoTML(X, A, Y, model_type_theta='nn', model_type_alpha='nn', n_splits=5, lambda_theta=0.001, lambda_alpha=0.001, l1_ratio=0.5,
           theta_epochs=200, alpha_epochs=300, batch_size_theta=128, batch_size_alpha=None, lr_theta=0.01, lr_alpha=0.01, 
           seed=None, alpha_nn_setup=None, run_id=None, pair_info=None):
    """
    Implement the autoTML algorithm with cross-fitting using specified model types
    
    Parameters:
    -----------
    X : numpy.ndarray
        Binary covariates (n_samples x p)
    A : numpy.ndarray
        Binary treatment assignment
    Y : numpy.ndarray
        Binary outcome
    model_type_theta : str
        Type of model to use for theta: 'linear', 'nn' (neural network), 'constant' (theta as constant), or 'X_only' (theta as main terms linear model of X only)
    model_type_alpha : str
        Type of model to use for alpha: 'linear', 'nn' (neural network), or 'polynomial' (polynomial with two-way interactions)
    n_splits : int
        Number of cross-fitting splits
    lambda_theta, lambda_alpha : float
        Regularization parameters
    theta_epochs, alpha_epochs : int
        Number of training epochs
    batch_size : int
        Batch size for training
    lr_theta, lr_alpha : float
        Learning rates
    seed : int, optional
        Random seed for reproducibility
    conv_threshold : float
        Convergence threshold for early stopping
    patience : int
        Number of epochs with small changes before early stopping
        
    Returns:
    --------
    dict
        Dictionary with estimation results
    """
    start_time = time.time()
    
    # Use the current device context set by the worker
    # Do NOT reinitialize GPU context here to avoid conflicts
    current_device = get_current_device()
    
    print(f"[DEBUG] autoTML: Started with device {current_device}")
    sys.stdout.flush()
    
    # Handle backward compatibility for batch sizes
    if batch_size_alpha is None:
        batch_size_alpha = batch_size_theta
    
    # Set seed if provided (for parallel runs)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    n = len(X)
    input_dim = X.shape[1]
    
    print(f"[DEBUG] autoTML: Data size n={n}, input_dim={input_dim}")
    sys.stdout.flush()
    
    # Convert data to PyTorch tensors using current device context
    print(f"[DEBUG] autoTML: Converting tensors to device {current_device}")
    sys.stdout.flush()
    
    X_tensor = torch.FloatTensor(X.copy()).to(current_device)
    A_tensor = torch.FloatTensor(A.copy()).to(current_device)
    Y_tensor = torch.FloatTensor(Y.copy()).to(current_device)
    
    # Step 0: Split data into 85% training and 15% validation
    print("Step 0: Splitting data into training (85%) and validation (15%) sets...")
    indices = np.arange(n)
    np.random.shuffle(indices)
    
    # Calculate split sizes
    n_val = int(0.15 * n)
    n_train = n - n_val
    
    # Split indices
    val_indices = indices[:n_val]
    train_indices = indices[n_val:]
    
    print(f"  Training set size: {n_train} samples")
    print(f"  Validation set size: {n_val} samples")
    
    # Extract training and validation data
    X_train_full = X_tensor[train_indices]
    A_train_full = A_tensor[train_indices]
    Y_train_full = Y_tensor[train_indices]
    
    X_val = X_tensor[val_indices]
    A_val = A_tensor[val_indices]
    Y_val = Y_tensor[val_indices]
    
    # Create data splits for cross-fitting on the training set only
    train_indices_shuffled = np.arange(n_train)
    np.random.shuffle(train_indices_shuffled)
    split_indices = np.array_split(train_indices_shuffled, n_splits)
    
    # Maps each training sample to its fold
    fold_mapping = np.zeros(n_train, dtype=int)
    for j, fold_indices in enumerate(split_indices):
        fold_mapping[fold_indices] = j
    
    # Initialize lists to store cross-fitted models and learning curves
    theta_models = []
    alpha_models = []
    theta_learning_curves = []
    alpha_learning_curves = []
    
    # Step 1: Cross-fit theta models
    print("Step 1: Cross-fitting theta models...")
    theta_training_start = time.time()  # Start timing theta training
    for s in range(n_splits):
        print(f"Training theta model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Training data for this split
        X_train = X_train_full[train_indices]
        A_train = A_train_full[train_indices]
        Y_train = Y_train_full[train_indices]
        
        # Create and train theta model
        theta_model = get_theta_model(model_type_theta, input_dim)
        theta_model, theta_losses = train_theta(theta_model, A_train, X_train, Y_train, 
                               lambda_theta=lambda_theta, epochs=theta_epochs, 
                               batch_size=batch_size_theta, lr=lr_theta,
                               fold_idx=s, run_id=run_id)
        
        # Store theta model and learning curve
        theta_models.append(theta_model)
        theta_learning_curves.append(theta_losses)
    
    theta_training_time = time.time() - theta_training_start  # Calculate theta training time
    # Step 2: Cross-fit alpha models using pre-trained theta models
    print("Step 2: Cross-fitting alpha models...")
    alpha_training_start = time.time()  # Start timing alpha training
    
    # PRE-COMPUTE: Calculate theta predictions and sigmoid derivatives for all training samples
    # This avoids repeated computation and device transfers during alpha training
    print("  Pre-computing theta predictions for alpha training...")
    theta_predictions = torch.zeros(n_train, device=current_device)
    sigmoid_derivatives = torch.zeros(n_train, device=current_device)
    
    # Pre-compute predictions using cross-fitted theta models
    for i in range(n_train):
        fold = fold_mapping[i]
        theta_model = theta_models[fold]
        theta_model.eval()
        
        with torch.no_grad():
            theta_pred = theta_model(A_train_full[i:i+1], X_train_full[i:i+1]).squeeze()
            sigmoid_theta = torch.sigmoid(theta_pred)
            sigmoid_deriv = sigmoid_theta * (1 - sigmoid_theta)
            
            theta_predictions[i] = theta_pred
            sigmoid_derivatives[i] = sigmoid_deriv
    
    for s in range(n_splits):
        print(f"Training alpha model for fold {s+1}/{n_splits}")
        
        # Get train and test indices for this split
        test_indices = split_indices[s]
        train_indices = np.concatenate([split_indices[j] for j in range(n_splits) if j != s])
        
        # Initialize alpha model
        alpha_model = get_alpha_model(model_type_alpha, input_dim, alpha_nn_setup)
        optimizer = optim.Adam(alpha_model.parameters(), lr=lr_alpha)
        
        # Create dataset for training with pre-computed theta info
        train_dataset = TensorDataset(
            torch.tensor(train_indices, dtype=torch.long, device=current_device),
            A_train_full[train_indices], 
            X_train_full[train_indices],
            theta_predictions[train_indices],
            sigmoid_derivatives[train_indices]
        )
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size_alpha, shuffle=True)
        
        # Training loop
        if hasattr(alpha_model, 'g0') and hasattr(alpha_model, 'g1'):
            alpha_model.g0.train()
            alpha_model.g1.train()
        
        # For learning curve tracking
        alpha_losses = []
        
        for epoch in range(alpha_epochs):
            epoch_total_loss = 0
            epoch_total_samples = 0
            
            for batch_indices, batch_a, batch_x, batch_theta_preds, batch_sigmoid_derivs in train_dataloader:
                optimizer.zero_grad(set_to_none=True)
                
                # Use pre-computed sigmoid derivatives (no theta model calls needed!)
                sigmoid_derivs = batch_sigmoid_derivs
                
                # Alpha for current (a,x) pairs
                alpha_axs = alpha_model(batch_a, batch_x).squeeze(-1)
                
                # Compute alpha(1,x) and alpha(0,x) for all x in this batch
                alpha_1xs = alpha_model.g1(batch_x).squeeze(-1)
                alpha_0xs = alpha_model.g0(batch_x).squeeze(-1)
                alpha_diffs = alpha_1xs - alpha_0xs
                
                # Loss terms (vectorized) - using pre-computed sigmoid derivatives
                first_terms = sigmoid_derivs * alpha_axs**2
                second_terms = -2 * alpha_diffs
                
                # Batch loss
                batch_losses = first_terms + second_terms
                batch_mean_loss = batch_losses.mean()
                
                # Add regularization (elastic net for polynomial, L1+L2 for NN, L2 for linear)
                reg_penalty = get_alpha_regularization(alpha_model, lambda_alpha, l1_ratio=l1_ratio, exclude_bias=True)
                loss = batch_mean_loss + reg_penalty
                
                # Backward and optimize
                loss.backward()
                optimizer.step()
                
                # Accumulate for correct averaging
                # Per-sample epoch average of the unregularised loss
                bs = batch_a.size(0)
                epoch_total_loss += batch_mean_loss.item() * bs
                epoch_total_samples += bs
            
            # Calculate correct average loss per sample for this epoch
            avg_epoch_loss = epoch_total_loss / epoch_total_samples if epoch_total_samples > 0 else epoch_total_loss
            alpha_losses.append(avg_epoch_loss)
        
        # Store alpha model and learning curve
        alpha_models.append(alpha_model)
        alpha_learning_curves.append(alpha_losses)
    
    alpha_training_time = time.time() - alpha_training_start  # Calculate alpha training time
    
    # Note: Learning curves will be plotted in the main process to ensure wandb compatibility
    
    # Step 2.5: Compute validation losses
    print("Step 2.5: Computing validation losses...")
    
    # Train an extra theta model using the entire training set
    print("  Training extra theta model on full training set...")
    extra_theta_model = get_theta_model(model_type_theta, input_dim)
    extra_theta_model, _ = train_theta(extra_theta_model, A_train_full, X_train_full, Y_train_full, 
                           lambda_theta=lambda_theta, epochs=theta_epochs, 
                           batch_size=batch_size_theta, lr=lr_theta,
                           fold_idx=-1, run_id=run_id)
    
    # Compute validation losses for each alpha model
    alpha_validation_losses = []
    
    for k in range(n_splits):
        print(f"  Computing validation loss for alpha model {k+1}/{n_splits}")
        alpha_model = alpha_models[k]
        
        # Compute theta predictions and sigmoid derivatives on validation set
        val_theta_predictions = torch.zeros(n_val, device=current_device)
        val_sigmoid_derivatives = torch.zeros(n_val, device=current_device)
        
        extra_theta_model.eval()
        with torch.no_grad():
            for i in range(n_val):
                theta_pred = extra_theta_model(A_val[i:i+1], X_val[i:i+1]).squeeze()
                sigmoid_theta = torch.sigmoid(theta_pred)
                sigmoid_deriv = sigmoid_theta * (1 - sigmoid_theta)
                
                val_theta_predictions[i] = theta_pred
                val_sigmoid_derivatives[i] = sigmoid_deriv
        
        # Compute alpha loss on validation set
        val_dataset = TensorDataset(A_val, X_val, val_theta_predictions, val_sigmoid_derivatives)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size_alpha, shuffle=False)
        
        total_val_loss = 0
        total_val_samples = 0
        
        alpha_model.g0.eval()
        alpha_model.g1.eval()
        
        with torch.no_grad():
            for batch_a, batch_x, batch_theta_preds, batch_sigmoid_derivs in val_dataloader:
                # Alpha for current (a,x) pairs
                alpha_axs = alpha_model(batch_a, batch_x).squeeze(-1)
                
                # Compute alpha(1,x) and alpha(0,x) for all x in this batch
                alpha_1xs = alpha_model.g1(batch_x).squeeze(-1)
                alpha_0xs = alpha_model.g0(batch_x).squeeze(-1)
                alpha_diffs = alpha_1xs - alpha_0xs
                
                # Loss terms (vectorized) - using pre-computed sigmoid derivatives
                first_terms = batch_sigmoid_derivs * alpha_axs**2
                second_terms = -2 * alpha_diffs
                
                # Batch loss (without regularization for validation)
                batch_losses = first_terms + second_terms
                batch_total_loss = batch_losses.sum()
                
                total_val_loss += batch_total_loss.item()
                total_val_samples += batch_a.size(0)
        
        avg_val_loss = total_val_loss / total_val_samples if total_val_samples > 0 else total_val_loss
        alpha_validation_losses.append(avg_val_loss)
        print(f"    Alpha model {k+1} validation loss: {avg_val_loss:.6f}")
    
    # Compute mean validation loss
    mean_alpha_validation_loss = np.mean(alpha_validation_losses)
    print(f"  Mean alpha validation loss: {mean_alpha_validation_loss:.6f}")
    
    # Step 3: Targeting step - compute epsilon_n
    # Collect theta and alpha predictions for all samples
    print("Step 3: Computing targeting step (epsilon_n)...")
    
    # Use larger batch size for evaluation
    batch_size_eval = min(5000, n_train)
    n_batches = int(np.ceil(n_train / batch_size_eval))
    
    theta_preds = np.zeros(n_train)
    alpha_preds = np.zeros(n_train)
    
    for batch in range(n_batches):
        start_idx = batch * batch_size_eval
        end_idx = min((batch + 1) * batch_size_eval, n_train)
        batch_indices = np.arange(start_idx, end_idx)
        
        for i in batch_indices:
            # Get the model fold for this sample
            fold = fold_mapping[i]
            
            # Get models trained without this sample
            theta_model = theta_models[fold]
            alpha_model = alpha_models[fold]
            
            # Get data for this sample
            x_i = X_train_full[i:i+1]
            a_i = A_train_full[i:i+1]
            
            with torch.no_grad():
                # Predict theta(A,X) and alpha(A,X)
                theta_preds[i] = theta_model(a_i, x_i).item()
                alpha_preds[i] = alpha_model(a_i, x_i).item()
    
    # Define negative log-likelihood loss function for epsilon optimization
    def neg_log_likelihood_loss(epsilon):
        # θ* = θ + ε·α
        theta_star = theta_preds + epsilon * alpha_preds
        
        # -Y·θ*(A,X) + log(1 + exp(θ*(A,X)))
        neg_ll = -Y_train_full.cpu().numpy() * theta_star + np.log(1 + np.exp(theta_star))
        return np.sum(neg_ll)

    # Optimize epsilon using scipy
    result = minimize(neg_log_likelihood_loss, x0=0, method='BFGS')
    epsilon_n = result.x[0]
    print(f"  Computed epsilon_n = {epsilon_n:.6f}")
    
    # Step 4: Update theta to theta* = theta + epsilon_n·alpha
    print("Step 4: Computing plug-in estimator...")
    
    # Initialize arrays to store efficient influence function values and EIF terms
    eif_values = np.zeros(n_train)
    eif_terms = np.zeros(n_train)
    
    # Compute plug-in estimator
    sum_term = 0
    for batch in range(n_batches):
        start_idx = batch * batch_size_eval
        end_idx = min((batch + 1) * batch_size_eval, n_train)
        batch_indices = np.arange(start_idx, end_idx)
        
        batch_sum = 0
        
        for i in batch_indices:
            # Get the model fold for this sample
            fold = fold_mapping[i]
            
            # Get models trained without this sample
            theta_model = theta_models[fold]
            alpha_model = alpha_models[fold]
            
            # Get data for this sample
            x_i = X_train_full[i:i+1]
            a_i = A_train_full[i:i+1]
            y_i = Y_train_full[i:i+1].item()
            
            with torch.no_grad():
                # Original theta predictions
                theta_a_x = theta_model(a_i, x_i).item()
                
                # Alpha prediction
                alpha_a_x = alpha_model(a_i, x_i).item()
                
                # Compute theta* = theta + epsilon_n·alpha
                theta_star_a_x = theta_a_x + epsilon_n * alpha_a_x
                
                # Predictions for a=1 and a=0
                theta_star_1_x = theta_model(torch.ones_like(a_i), x_i).item() + epsilon_n * alpha_model(torch.ones_like(a_i), x_i).item()
                theta_star_0_x = theta_model(torch.zeros_like(a_i), x_i).item() + epsilon_n * alpha_model(torch.zeros_like(a_i), x_i).item()
                
                # Plug-in estimate contribution: theta*(1,X) - theta*(0,X)
                plug_in_term = theta_star_1_x - theta_star_0_x
                
                # Store for the final estimator
                batch_sum += plug_in_term
                
                # Calculate efficient influence function component: (sigma(theta*(A,X)) - Y)·alpha(A,X)
                sigma_theta_star = sigmoid(theta_star_a_x)
                eif_term = (sigma_theta_star - y_i) * alpha_a_x
                
                # Store EIF terms for reporting
                eif_terms[i] = eif_term
                
                # Store EIF value for variance calculation
                eif_values[i] = plug_in_term - eif_term
        
        sum_term += batch_sum
    
    # Final plug-in estimate
    psi_hat = sum_term / n_train
    
    # Calculate average EIF term (should be close to zero for well-specified models)
    avg_eif_term = np.mean(eif_terms)
    
    # Step 5: Compute standard error
    print("Step 5: Computing standard error...")
    
    # Calculate variance using the EIF values
    centered_eif = eif_values - psi_hat
    V_hat = np.mean(centered_eif**2)
    
    # Standard error
    se_hat = np.sqrt(V_hat / n_train)
    
    # 95% confidence interval
    ci_lower = psi_hat - 1.96 * se_hat
    ci_upper = psi_hat + 1.96 * se_hat
    
    # Calculate average EIF value (should be close to 0 for well-specified models)
    avg_eif = np.mean(eif_values - psi_hat)
    
    # Total execution time
    execution_time = time.time() - start_time
    
    # Return results
    results = {
        'psi_hat': psi_hat,
        'se_hat': se_hat,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'variance': V_hat,
        'epsilon_n': epsilon_n,
        'avg_eif': avg_eif,
        'avg_eif_term': avg_eif_term,
        'model_type_theta': model_type_theta,
        'model_type_alpha': model_type_alpha,
        'execution_time': execution_time,
        'theta_training_time': theta_training_time,
        'alpha_training_time': alpha_training_time,
        'theta_learning_curves': theta_learning_curves,
        'alpha_learning_curves': alpha_learning_curves,
        'alpha_validation_losses': alpha_validation_losses,
        'mean_alpha_validation_loss': mean_alpha_validation_loss
    }
    
    return results

# Main wrapper function for flexible use
def debiased_binary_estimator(data, y_index, a_index, method='autoDML', model_type_theta='linear', model_type_alpha='linear', 
                              stabilization=True, n_splits=5, lambda_theta=0.001, lambda_alpha=0.001, l1_ratio=0.5,
                              theta_epochs=200, alpha_epochs=300, batch_size_theta=128, batch_size_alpha=None,
                              lr_theta=0.01, lr_alpha=0.01, seed=None, alpha_nn_setup=None,
                              run_id=None, pair_info=None):
    """
    Flexible wrapper for debiased estimation methods on binary data
    
    Parameters:
    -----------
    data : numpy.ndarray
        Binary data with shape (n_samples, n_variables)
    y_index : int
        Index of the outcome variable Y in the data
    a_index : int
        Index of the treatment variable A in the data
    method : str, optional
        Estimation method, either 'autoDML' or 'autoTML' (default: 'autoDML')
    model_type_theta : str, optional
        Type of model to use for theta: 'linear', 'nn' (neural network), 'constant' (theta as constant), or 'X_only' (theta as main terms linear model of X only) (default: 'linear')
    model_type_alpha : str, optional
        Type of model to use for alpha: 'linear', 'nn' (neural network), or 'polynomial' (polynomial with two-way interactions) (default: 'linear')
    stabilization : bool, optional
        Whether to apply TMLE-inspired stabilization in autoDML (default: True)
    n_splits : int, optional
        Number of cross-fitting splits (default: 5)
    lambda_theta, lambda_alpha : float, optional
        Regularization parameters (default: 0.001)
    theta_epochs, alpha_epochs : int, optional
        Number of training epochs (default: 300)
    batch_size : int, optional
        Batch size for training (default: 64)
    lr_theta, lr_alpha : float, optional
        Learning rates (default: 0.01)
    seed : int, optional
        Random seed for reproducibility
    conv_threshold : float, optional
        Convergence threshold for early stopping (default: 1e-5)
    patience : int, optional
        Number of epochs with small changes before early stopping (default: 20)
        
    Returns:
    --------
    dict
        Dictionary with estimation results
    """
    # Use the current device context set by the worker
    # Do NOT reinitialize GPU context here to avoid conflicts
    current_device = get_current_device()
    
    print(f"[DEBUG] debiased_binary_estimator: Started for pair ({y_index}, {a_index})")
    print(f"[DEBUG] Current device: {current_device}")
    sys.stdout.flush()
    
    try:
        # Validate inputs
        if not isinstance(data, np.ndarray):
            try:
                data = np.array(data)
            except:
                raise ValueError("data must be convertible to a numpy array")
        
        print(f"[DEBUG] Input validation passed")
        sys.stdout.flush()
        
        n, p = data.shape
        
        # Check if indices are within valid range
        if y_index < 0 or y_index >= p:
            raise ValueError(f"y_index must be between 0 and {p-1}")
        if a_index < 0 or a_index >= p:
            raise ValueError(f"a_index must be between 0 and {p-1}")
        if y_index == a_index:
            raise ValueError("y_index and a_index cannot be the same")
        
        print(f"[DEBUG] Index validation passed")
        sys.stdout.flush()
        
        # Extract Y, A, and X
        Y = data[:, y_index]
        A = data[:, a_index]
        
        # Get all column indices except y_index and a_index
        X_indices = [i for i in range(p) if i != y_index and i != a_index]
        X = data[:, X_indices]
        
        # Handle backward compatibility for batch sizes
        if batch_size_alpha is None:
            batch_size_alpha = batch_size_theta
        
        # Check if data is binary (0, 1)
        if not np.all(np.isin(Y, [0, 1])):
            raise ValueError("Y must contain only binary values (0, 1)")
        if not np.all(np.isin(A, [0, 1])):
            raise ValueError("A must contain only binary values (0, 1)")
        
        # Validate method and model_type
        if method not in ['autoDML', 'autoTML']:
            raise ValueError("method must be either 'autoDML' or 'autoTML'")
        if model_type_theta not in ['linear', 'nn', 'constant', 'X_only']:
            raise ValueError("model_type_theta must be either 'linear', 'nn', 'constant', or 'X_only'")
        if model_type_alpha not in ['linear', 'nn', 'polynomial']:
            raise ValueError("model_type_alpha must be either 'linear', 'nn', or 'polynomial'")
        
        # Print information
        print(f"Dataset shape: {n} samples × {p} variables")
        print(f"Y variable (outcome) index: {y_index}")
        print(f"A variable (treatment) index: {a_index}")
        print(f"X variables (covariates): {len(X_indices)} features")
        print(f"Method: {method} with theta={model_type_theta}, alpha={model_type_alpha} models")
        print(f"Cross-fitting splits: {n_splits}")
        print(f"Using device: {current_device}")
        if method == 'autoDML':
            print(f"Stabilization: {stabilization}")
        
        print(f"[DEBUG] About to call {method} algorithm")
        sys.stdout.flush()
        
        # Run estimation with current device context
        results = None
        if method == 'autoDML':
            results = autoDML(X, A, Y, model_type_theta=model_type_theta, model_type_alpha=model_type_alpha, n_splits=n_splits,
                            lambda_theta=lambda_theta, lambda_alpha=lambda_alpha, l1_ratio=l1_ratio,
                            theta_epochs=theta_epochs, alpha_epochs=alpha_epochs,
                            batch_size_theta=batch_size_theta, batch_size_alpha=batch_size_alpha, lr_theta=lr_theta, lr_alpha=lr_alpha,
                            stabilization=stabilization, seed=seed, alpha_nn_setup=alpha_nn_setup,
                            run_id=run_id, pair_info=pair_info)
        elif method == 'autoTML':
            results = autoTML(X, A, Y, model_type_theta=model_type_theta, model_type_alpha=model_type_alpha, n_splits=n_splits,
                             lambda_theta=lambda_theta, lambda_alpha=lambda_alpha, l1_ratio=l1_ratio,
                             theta_epochs=theta_epochs, alpha_epochs=alpha_epochs,
                             batch_size_theta=batch_size_theta, batch_size_alpha=batch_size_alpha, lr_theta=lr_theta, lr_alpha=lr_alpha,
                             seed=seed, alpha_nn_setup=alpha_nn_setup,
                             run_id=run_id, pair_info=pair_info)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Add additional information to results
        if results is not None:
            results['method'] = method
            results['y_index'] = y_index
            results['a_index'] = a_index
            results['n_samples'] = n
            results['n_variables'] = p
            results['n_covariates'] = len(X_indices)
        
        return results
    
    except Exception as e:
        # Safe cleanup that doesn't interfere with other workers
        if torch.cuda.is_available():
            torch.cuda.synchronize()  # Only sync, don't clear cache
        raise e

# Function to print results in a nice format
def print_results(results):
    """Print debiased estimation results in a formatted way"""
    print("\n" + "="*50)
    print(f"DEBIASED ESTIMATION RESULTS: {results['method'].upper()}")
    print("="*50)
    
    # Handle both old and new model type formats for backward compatibility
    if 'model_type_theta' in results and 'model_type_alpha' in results:
        print(f"Theta model type: {results['model_type_theta']}")
        print(f"Alpha model type: {results['model_type_alpha']}")
    elif 'model_type' in results:
        print(f"Model type: {results['model_type']}")
    
    if 'stabilization' in results:
        print(f"Stabilization: {results['stabilization']}")
    print("\nESTIMATES:")
    print(f"ATE (point estimate): {results['psi_hat']:.6f}")
    print(f"Standard error: {results['se_hat']:.6f}")
    print(f"95% CI: [{results['ci_lower']:.6f}, {results['ci_upper']:.6f}]")
    
    print("\nDATASET INFO:")
    print(f"Number of samples: {results['n_samples']}")
    print(f"Total variables: {results['n_variables']}")
    print(f"Y variable index: {results['y_index']}")
    print(f"A variable index: {results['a_index']}")
    print(f"Number of covariates: {results['n_covariates']}")
    
    if 'epsilon_n' in results:
        print("\nDIAGNOSTICS:")
        print(f"Targeting step epsilon: {results['epsilon_n']:.6f}")
        print(f"Average EIF term: {results['avg_eif_term']:.6f}")
    elif 'stabilization_factor' in results and results['stabilization_factor'] is not None:
        print("\nDIAGNOSTICS:")
        print(f"Stabilization factor: {results['stabilization_factor']:.6f}")
    
    print(f"\nExecution time: {results['execution_time']:.2f} seconds")
    print("="*50)