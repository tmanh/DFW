"""
Cokriging Implementation
========================
This module implements Ordinary Cokriging for spatial interpolation using
multiple correlated variables to improve predictions.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
import matplotlib.pyplot as plt
from typing import Tuple, Optional, List, Dict


class OrdinaryCokriging:
    """
    Ordinary Cokriging implementation.
    
    Cokriging uses multiple correlated spatial variables jointly to make
    predictions. The primary variable is predicted using both its own spatial
    correlation structure and cross-correlations with secondary variables.
    
    Parameters
    ----------
    variogram_models : dict
        Variogram models for each variable and cross-variograms.
        Keys should be: 'var0' (primary), 'var1', 'var2', ... (secondary),
        and 'cross_01', 'cross_02', etc. for cross-variograms.
        Values are tuples: (model_type, params)
        Example: {'var0': ('spherical', {'sill': 1.0, 'range': 10.0, 'nugget': 0.1})}
    n_variables : int
        Number of variables (primary + secondary)
    """
    
    def __init__(self, variogram_models: Dict[str, Tuple[str, dict]], 
                 n_variables: int = 2):
        self.variogram_models = variogram_models
        self.n_variables = n_variables
        
        # Fitted data
        self.X_train = {}  # Dictionary of training coordinates for each variable
        self.y_train = {}  # Dictionary of training values for each variable
        self.n_samples = {}  # Number of samples per variable
        
    def _variogram(self, h: np.ndarray, var_key: str) -> np.ndarray:
        """
        Calculate variogram value for distance h.
        
        Parameters
        ----------
        h : np.ndarray
            Distance array
        var_key : str
            Key for variogram model (e.g., 'var0', 'cross_01')
            
        Returns
        -------
        gamma : np.ndarray
            Variogram values
        """
        if var_key not in self.variogram_models:
            raise ValueError(f"Variogram model for '{var_key}' not found")
        
        model_type, params = self.variogram_models[var_key]
        sill = params.get('sill', 1.0)
        range_ = params.get('range', 10.0)
        nugget = params.get('nugget', 0.0)
        
        if callable(model_type):
            return model_type(h, **params)
        
        if model_type == 'spherical':
            gamma = np.where(
                h <= range_,
                nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
                sill
            )
        elif model_type == 'exponential':
            gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))
        elif model_type == 'gaussian':
            gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * (h / range_) ** 2))
        elif model_type == 'linear':
            # Linear model with sill as slope
            gamma = nugget + sill * h
        else:
            raise ValueError(f"Unknown variogram model: {model_type}")
        
        return gamma
    
    def fit(self, X_list: List[np.ndarray], y_list: List[np.ndarray]) -> 'OrdinaryCokriging':
        """
        Fit the Cokriging model with training data.
        
        Parameters
        ----------
        X_list : list of np.ndarray
            List of coordinate arrays for each variable.
            X_list[0] is primary variable, X_list[1:] are secondary variables.
            Each array has shape (n_samples_i, n_features)
        y_list : list of np.ndarray
            List of value arrays for each variable.
            y_list[0] is primary variable, y_list[1:] are secondary variables.
            Each array has shape (n_samples_i,)
            
        Returns
        -------
        self : OrdinaryCokriging
            Fitted model
        """
        if len(X_list) != self.n_variables or len(y_list) != self.n_variables:
            raise ValueError(f"Expected {self.n_variables} variables, got {len(X_list)}")
        
        for i in range(self.n_variables):
            self.X_train[i] = np.atleast_2d(X_list[i])
            self.y_train[i] = np.array(y_list[i])
            self.n_samples[i] = len(y_list[i])
        
        return self
    
    def _build_cokriging_matrix(self, X_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the cokriging system matrices for prediction.
        
        Parameters
        ----------
        X_pred : np.ndarray, shape (n_pred, n_features)
            Coordinates where predictions are needed (for primary variable)
            
        Returns
        -------
        A : np.ndarray
            Left-hand side cokriging matrix
        b : np.ndarray
            Right-hand side cokriging matrix
        """
        n_pred = X_pred.shape[0]
        
        # Calculate total number of data points and constraints
        total_samples = sum(self.n_samples.values())
        matrix_size = total_samples + self.n_variables
        
        # Initialize cokriging matrix
        A = np.zeros((matrix_size, matrix_size))
        
        # Build block structure
        # A = | C   | U |
        #     | U^T | 0 |
        # where C contains all auto- and cross-variograms
        # and U contains the unbiasedness constraints (ones)
        
        row_offset = 0
        col_offset = 0
        
        # Fill the covariance blocks
        for i in range(self.n_variables):
            for j in range(self.n_variables):
                n_i = self.n_samples[i]
                n_j = self.n_samples[j]
                
                # Calculate distances
                dist = cdist(self.X_train[i], self.X_train[j])
                
                # Get appropriate variogram
                if i == j:
                    var_key = f'var{i}'
                else:
                    # Cross-variogram (symmetric)
                    var_key = f'cross_{min(i,j)}{max(i,j)}'
                
                gamma = self._variogram(dist, var_key)
                
                # Fill block
                A[row_offset:row_offset+n_i, col_offset:col_offset+n_j] = gamma
                
                col_offset += n_j
            
            col_offset = 0
            row_offset += n_i
        
        # Add unbiasedness constraints (ones for each variable)
        constraint_offset = total_samples
        sample_offset = 0
        
        for i in range(self.n_variables):
            n_i = self.n_samples[i]
            A[sample_offset:sample_offset+n_i, constraint_offset+i] = 1
            A[constraint_offset+i, sample_offset:sample_offset+n_i] = 1
            sample_offset += n_i
        
        # Add small regularization to diagonal for numerical stability
        eps = 1e-6
        A[:total_samples, :total_samples] += eps * np.eye(total_samples)
        
        # Build right-hand side b for each prediction point
        b = np.zeros((n_pred, matrix_size))
        
        sample_offset = 0
        for i in range(self.n_variables):
            n_i = self.n_samples[i]
            
            # Distance from prediction points to training points of variable i
            if i == 0:
                # Primary variable - use prediction locations
                dist_pred = cdist(X_pred, self.X_train[i])
                var_key = 'var0'
            else:
                # Secondary variables - use cross-variogram with primary
                dist_pred = cdist(X_pred, self.X_train[i])
                var_key = f'cross_0{i}'
            
            gamma_pred = self._variogram(dist_pred, var_key)
            b[:, sample_offset:sample_offset+n_i] = gamma_pred
            
            sample_offset += n_i
        
        # Unbiasedness constraint for primary variable only
        b[:, total_samples] = 1
        
        return A, b
    
    def predict(self, X_pred: np.ndarray, 
                return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict primary variable values at new locations using Cokriging.
        
        Parameters
        ----------
        X_pred : np.ndarray, shape (n_pred, n_features)
            Coordinates where predictions are needed
        return_std : bool, default=False
            Whether to return kriging standard deviation
            
        Returns
        -------
        y_pred : np.ndarray, shape (n_pred,)
            Predicted values for primary variable
        sigma : np.ndarray, shape (n_pred,), optional
            Kriging standard deviation (if return_std=True)
        """
        if not self.X_train:
            raise ValueError("Model must be fitted before prediction")
        
        X_pred = np.atleast_2d(X_pred)
        
        # Build cokriging system
        A, b = self._build_cokriging_matrix(X_pred)
        
        # Solve cokriging system for weights
        weights = _solve_cokriging(A, b)
        
        # Calculate predictions using all variables
        y_pred = np.zeros(X_pred.shape[0])
        
        sample_offset = 0
        for i in range(self.n_variables):
            n_i = self.n_samples[i]
            y_pred += weights[:, sample_offset:sample_offset+n_i] @ self.y_train[i]
            sample_offset += n_i
        
        if return_std:
            # Calculate cokriging variance
            sigma_squared = np.sum(weights * b, axis=1)
            sigma = np.sqrt(np.maximum(sigma_squared, 0))  # Ensure non-negative
            return y_pred, sigma
        
        return y_pred, None


def _solve_cokriging(A, b):
    """
    Solve cokriging system with robust fallbacks.

    Parameters
    ----------
    A : ndarray, shape (M, M)
        Cokriging matrix
    b : ndarray, shape (n_pred, M)
        RHS matrix (one row per prediction point)

    Returns
    -------
    weights : ndarray, shape (n_pred, M)
    """
    B = b.T  # (M, n_pred)

    try:
        # 1) Exact solve
        W = solve(A, B)
    except np.linalg.LinAlgError:
        try:
            # 2) Least squares fallback
            W = np.linalg.lstsq(A, B, rcond=1e-10)[0]
        except Exception:
            # 3) Regularized solve (last resort)
            eps = 1e-5
            A_reg = A + eps * np.eye(A.shape[0])
            W = solve(A_reg, B)

    return W.T  # (n_pred, M)


# Example usage and demonstration
def example_1d():
    """Simple 1D example with synthetic data"""
    print("=" * 60)
    print("Example 1: 1D Ordinary Cokriging")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    
    # Primary variable (sparse sampling)
    n_primary = 15
    X_primary = np.linspace(0, 100, n_primary).reshape(-1, 1)
    y_primary = (20 * np.sin(X_primary.ravel() / 15) + 
                 10 * np.cos(X_primary.ravel() / 25) +
                 np.random.normal(0, 2, n_primary))
    
    # Secondary variable (dense sampling, correlated with primary)
    n_secondary = 40
    X_secondary = np.linspace(0, 100, n_secondary).reshape(-1, 1)
    y_secondary = (15 * np.sin(X_secondary.ravel() / 15) + 
                   8 * np.cos(X_secondary.ravel() / 25) +
                   5 * np.sin(X_secondary.ravel() / 8) +
                   np.random.normal(0, 1.5, n_secondary))
    
    # Prediction points
    X_pred = np.linspace(0, 100, 200).reshape(-1, 1)
    
    # Define variogram models
    variogram_models = {
        'var0': ('spherical', {'sill': 40.0, 'range': 30.0, 'nugget': 2.0}),
        'var1': ('spherical', {'sill': 30.0, 'range': 25.0, 'nugget': 1.5}),
        'cross_01': ('spherical', {'sill': 25.0, 'range': 28.0, 'nugget': 1.0})
    }
    
    # Fit Cokriging model
    cok = OrdinaryCokriging(variogram_models=variogram_models, n_variables=2)
    cok.fit([X_primary, X_secondary], [y_primary, y_secondary])
    
    # Predict
    y_pred, sigma = cok.predict(X_pred, return_std=True)
    
    # Plot results
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_primary, y_primary, c='red', s=80, label='Primary variable', 
                zorder=3, edgecolors='darkred', linewidth=1.5)
    plt.scatter(X_secondary, y_secondary, c='blue', s=30, alpha=0.5, 
                label='Secondary variable', zorder=2)
    plt.plot(X_pred, y_pred, 'g-', label='Cokriging prediction', linewidth=2.5)
    plt.fill_between(X_pred.ravel(), 
                     y_pred - 2*sigma, 
                     y_pred + 2*sigma, 
                     alpha=0.3, color='green', label='±2σ confidence')
    plt.xlabel('Location', fontsize=11)
    plt.ylabel('Value', fontsize=11)
    plt.title('Ordinary Cokriging - 1D Predictions', fontsize=12, fontweight='bold')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(X_pred, sigma, 'purple', linewidth=2)
    plt.axhline(y=np.mean(sigma), color='orange', linestyle='--', 
                label=f'Mean σ = {np.mean(sigma):.2f}')
    plt.xlabel('Location', fontsize=11)
    plt.ylabel('Prediction Std Dev (σ)', fontsize=11)
    plt.title('Prediction Uncertainty', fontsize=12, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/cokriging_1d_example.png', dpi=150, bbox_inches='tight')
    print("✓ 1D example plot saved")
    
    return cok, y_pred, sigma


def example_2d():
    """2D spatial example with multiple variables"""
    print("\n" + "=" * 60)
    print("Example 2: 2D Ordinary Cokriging")
    print("=" * 60)
    
    # Generate 2D synthetic data
    np.random.seed(42)
    
    # Primary variable (sparse sampling - e.g., soil moisture)
    n_primary = 30
    X_primary = np.random.uniform(0, 100, (n_primary, 2))
    y_primary = (20 * np.sin(X_primary[:, 0] / 20) * np.cos(X_primary[:, 1] / 20) +
                 0.3 * X_primary[:, 0] + 0.2 * X_primary[:, 1] +
                 np.random.normal(0, 3, n_primary))
    
    # Secondary variable (denser sampling - e.g., temperature)
    n_secondary = 60
    X_secondary = np.random.uniform(0, 100, (n_secondary, 2))
    y_secondary = (15 * np.sin(X_secondary[:, 0] / 20) * np.cos(X_secondary[:, 1] / 20) +
                   0.25 * X_secondary[:, 0] + 0.15 * X_secondary[:, 1] +
                   8 * np.sin(X_secondary[:, 0] / 15) +
                   np.random.normal(0, 2, n_secondary))
    
    # Create prediction grid
    x_grid = np.linspace(0, 100, 50)
    y_grid = np.linspace(0, 100, 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Define variogram models
    variogram_models = {
        'var0': ('exponential', {'sill': 80.0, 'range': 35.0, 'nugget': 4.0}),
        'var1': ('exponential', {'sill': 60.0, 'range': 30.0, 'nugget': 3.0}),
        'cross_01': ('exponential', {'sill': 50.0, 'range': 32.0, 'nugget': 2.0})
    }
    
    # Fit Cokriging model
    cok = OrdinaryCokriging(variogram_models=variogram_models, n_variables=2)
    cok.fit([X_primary, X_secondary], [y_primary, y_secondary])
    
    # Predict
    y_pred, sigma = cok.predict(X_pred, return_std=True)
    
    # Reshape for plotting
    Z_pred = y_pred.reshape(X_grid.shape)
    Z_sigma = sigma.reshape(X_grid.shape)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    
    # Prediction map
    im1 = axes[0].contourf(X_grid, Y_grid, Z_pred, levels=20, cmap='viridis')
    axes[0].scatter(X_primary[:, 0], X_primary[:, 1], c=y_primary, 
                   s=100, edgecolors='white', linewidth=2, cmap='viridis',
                   label='Primary var')
    axes[0].scatter(X_secondary[:, 0], X_secondary[:, 1], c='red', 
                   s=20, alpha=0.4, marker='^', label='Secondary var')
    axes[0].set_xlabel('X coordinate', fontsize=11)
    axes[0].set_ylabel('Y coordinate', fontsize=11)
    axes[0].set_title('Cokriging Predictions', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right')
    plt.colorbar(im1, ax=axes[0], label='Predicted value')
    
    # Uncertainty map
    im2 = axes[1].contourf(X_grid, Y_grid, Z_sigma, levels=20, cmap='Reds')
    axes[1].scatter(X_primary[:, 0], X_primary[:, 1], c='blue', 
                   s=80, alpha=0.7, edgecolors='white', linewidth=1)
    axes[1].scatter(X_secondary[:, 0], X_secondary[:, 1], c='green', 
                   s=20, alpha=0.4, marker='^')
    axes[1].set_xlabel('X coordinate', fontsize=11)
    axes[1].set_ylabel('Y coordinate', fontsize=11)
    axes[1].set_title('Prediction Uncertainty (σ)', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Standard deviation')
    
    # Sampling density comparison
    axes[2].scatter(X_primary[:, 0], X_primary[:, 1], c='red', 
                   s=100, alpha=0.7, label=f'Primary (n={n_primary})', 
                   edgecolors='darkred', linewidth=1.5)
    axes[2].scatter(X_secondary[:, 0], X_secondary[:, 1], c='blue', 
                   s=50, alpha=0.5, label=f'Secondary (n={n_secondary})',
                   marker='^')
    axes[2].set_xlabel('X coordinate', fontsize=11)
    axes[2].set_ylabel('Y coordinate', fontsize=11)
    axes[2].set_title('Sampling Locations', fontsize=12, fontweight='bold')
    axes[2].legend(loc='upper right')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/cokriging_2d_example.png', dpi=150, bbox_inches='tight')
    print("✓ 2D example plot saved")
    
    return cok, Z_pred, Z_sigma


def example_3_variables():
    """Example with 3 variables (1 primary + 2 secondary)"""
    print("\n" + "=" * 60)
    print("Example 3: Cokriging with 3 Variables")
    print("=" * 60)
    
    np.random.seed(42)
    
    # Primary variable (very sparse)
    n_primary = 10
    X_primary = np.random.uniform(0, 100, (n_primary, 2))
    y_primary = (25 * np.sin(X_primary[:, 0] / 18) +
                 0.4 * X_primary[:, 1] +
                 np.random.normal(0, 3, n_primary))
    
    # Secondary variable 1 (moderate sampling)
    n_sec1 = 30
    X_sec1 = np.random.uniform(0, 100, (n_sec1, 2))
    y_sec1 = (20 * np.sin(X_sec1[:, 0] / 18) +
              0.35 * X_sec1[:, 1] +
              5 * np.cos(X_sec1[:, 0] / 25) +
              np.random.normal(0, 2, n_sec1))
    
    # Secondary variable 2 (dense sampling)
    n_sec2 = 50
    X_sec2 = np.random.uniform(0, 100, (n_sec2, 2))
    y_sec2 = (18 * np.sin(X_sec2[:, 0] / 18) +
              0.3 * X_sec2[:, 1] +
              8 * np.sin(X_sec2[:, 1] / 20) +
              np.random.normal(0, 2.5, n_sec2))
    
    # Prediction grid
    x_grid = np.linspace(0, 100, 40)
    y_grid = np.linspace(0, 100, 40)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    # Define variogram models for 3 variables
    variogram_models = {
        'var0': ('spherical', {'sill': 70.0, 'range': 40.0, 'nugget': 3.0}),
        'var1': ('spherical', {'sill': 60.0, 'range': 35.0, 'nugget': 2.5}),
        'var2': ('spherical', {'sill': 65.0, 'range': 38.0, 'nugget': 3.5}),
        'cross_01': ('spherical', {'sill': 55.0, 'range': 37.0, 'nugget': 2.0}),
        'cross_02': ('spherical', {'sill': 50.0, 'range': 36.0, 'nugget': 2.5}),
        'cross_12': ('spherical', {'sill': 45.0, 'range': 34.0, 'nugget': 2.0})
    }
    
    # Fit Cokriging model with 3 variables
    cok = OrdinaryCokriging(variogram_models=variogram_models, n_variables=3)
    cok.fit([X_primary, X_sec1, X_sec2], [y_primary, y_sec1, y_sec2])
    
    # Predict
    y_pred, sigma = cok.predict(X_pred, return_std=True)
    
    # Reshape for plotting
    Z_pred = y_pred.reshape(X_grid.shape)
    Z_sigma = sigma.reshape(X_grid.shape)
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Prediction map
    im1 = axes[0].contourf(X_grid, Y_grid, Z_pred, levels=20, cmap='plasma')
    axes[0].scatter(X_primary[:, 0], X_primary[:, 1], c='red', 
                   s=150, edgecolors='white', linewidth=2, 
                   label=f'Primary (n={n_primary})', marker='o', zorder=5)
    axes[0].scatter(X_sec1[:, 0], X_sec1[:, 1], c='yellow', 
                   s=60, alpha=0.6, label=f'Secondary 1 (n={n_sec1})', 
                   marker='s', edgecolors='orange')
    axes[0].scatter(X_sec2[:, 0], X_sec2[:, 1], c='cyan', 
                   s=30, alpha=0.5, label=f'Secondary 2 (n={n_sec2})', 
                   marker='^')
    axes[0].set_xlabel('X coordinate', fontsize=11)
    axes[0].set_ylabel('Y coordinate', fontsize=11)
    axes[0].set_title('3-Variable Cokriging Predictions', fontsize=12, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    plt.colorbar(im1, ax=axes[0], label='Predicted value')
    
    # Uncertainty map
    im2 = axes[1].contourf(X_grid, Y_grid, Z_sigma, levels=20, cmap='YlOrRd')
    axes[1].scatter(X_primary[:, 0], X_primary[:, 1], c='blue', 
                   s=100, alpha=0.8, edgecolors='darkblue', linewidth=1.5)
    axes[1].set_xlabel('X coordinate', fontsize=11)
    axes[1].set_ylabel('Y coordinate', fontsize=11)
    axes[1].set_title('Prediction Uncertainty', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[1], label='Standard deviation')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/cokriging_3var_example.png', dpi=150, bbox_inches='tight')
    print("✓ 3-variable example plot saved")
    
    # Print statistics
    print(f"\nPrediction Statistics:")
    print(f"  Mean prediction: {np.mean(y_pred):.2f}")
    print(f"  Std of predictions: {np.std(y_pred):.2f}")
    print(f"  Mean uncertainty (σ): {np.mean(sigma):.2f}")
    print(f"  Max uncertainty: {np.max(sigma):.2f}")
    
    return cok, Z_pred, Z_sigma


if __name__ == "__main__":
    print("\nOrdinary Cokriging - Implementation Demo\n")
    
    # Run examples
    cok_1d, pred_1d, std_1d = example_1d()
    cok_2d, pred_2d, std_2d = example_2d()
    cok_3var, pred_3var, std_3var = example_3_variables()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("Plots saved to outputs directory:")
    print("  - cokriging_1d_example.png")
    print("  - cokriging_2d_example.png")
    print("  - cokriging_3var_example.png")
    print("=" * 60)
