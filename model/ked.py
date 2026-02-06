"""
Kriging with External Drift (KED) Implementation
================================================
This module implements Kriging with External Drift for spatial interpolation
using auxiliary/external variables to improve predictions.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.linalg import solve
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Callable


class KrigingExternalDrift:
    """
    Kriging with External Drift (KED) implementation.
    
    KED extends ordinary kriging by incorporating external drift variables
    that are correlated with the primary variable of interest.
    
    Parameters
    ----------
    variogram_model : str or callable
        Variogram model type: 'spherical', 'exponential', 'gaussian', or custom function
    variogram_params : dict
        Parameters for the variogram model (e.g., {'sill': 1.0, 'range': 10.0, 'nugget': 0.1})
    """
    
    def __init__(self, variogram_model: str = 'spherical', 
                 variogram_params: Optional[dict] = None):
        self.variogram_model = variogram_model
        self.variogram_params = variogram_params or {'sill': 1.0, 'range': 10.0, 'nugget': 0.0}
        
        # Fitted data
        self.X_train = None
        self.y_train = None
        self.drift_train = None
        self.n_samples = None
        self.n_drift = None
        
    def _variogram(self, h: np.ndarray) -> np.ndarray:
        """
        Calculate variogram value for distance h.
        
        Parameters
        ----------
        h : np.ndarray
            Distance array
            
        Returns
        -------
        gamma : np.ndarray
            Variogram values
        """
        sill = self.variogram_params.get('sill', 1.0)
        range_ = self.variogram_params.get('range', 10.0)
        nugget = self.variogram_params.get('nugget', 0.0)
        
        if callable(self.variogram_model):
            return self.variogram_model(h, **self.variogram_params)
        
        if self.variogram_model == 'spherical':
            gamma = np.where(
                h <= range_,
                nugget + (sill - nugget) * (1.5 * h / range_ - 0.5 * (h / range_) ** 3),
                sill
            )
        elif self.variogram_model == 'exponential':
            gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * h / range_))
        elif self.variogram_model == 'gaussian':
            gamma = nugget + (sill - nugget) * (1 - np.exp(-3 * (h / range_) ** 2))
        else:
            raise ValueError(f"Unknown variogram model: {self.variogram_model}")
        
        return gamma
    
    def fit(self, X: np.ndarray, y: np.ndarray, drift: np.ndarray) -> 'KrigingExternalDrift':
        """
        Fit the KED model with training data.
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Coordinates of known points (e.g., [longitude, latitude])
        y : np.ndarray, shape (n_samples,)
            Values at known points (primary variable)
        drift : np.ndarray, shape (n_samples,) or (n_samples, n_drift_vars)
            External drift variables at known points
            
        Returns
        -------
        self : KrigingExternalDrift
            Fitted model
        """
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        # Ensure drift is 2D
        drift_array = np.atleast_1d(drift)
        if drift_array.ndim == 1:
            self.drift_train = drift_array.reshape(-1, 1)
        else:
            self.drift_train = drift_array
        
        self.n_samples = len(y)
        self.n_drift = self.drift_train.shape[1]
        
        return self
    
    def _build_kriging_matrix(self, X_pred: np.ndarray, drift_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Build the kriging system matrices for prediction.
        
        Parameters
        ----------
        X_pred : np.ndarray, shape (n_pred, n_features)
            Coordinates where predictions are needed
        drift_pred : np.ndarray, shape (n_pred, n_drift_vars)
            External drift values at prediction points
            
        Returns
        -------
        A : np.ndarray
            Left-hand side kriging matrix
        b : np.ndarray
            Right-hand side kriging matrix
        """
        n_pred = X_pred.shape[0]
        
        # Calculate distances between training points
        dist_train = cdist(self.X_train, self.X_train)
        gamma_train = self._variogram(dist_train)
        
        # Build the kriging matrix A
        # Structure:
        # | Gamma    F     |
        # | F^T      0     |
        # where F contains drift variables and constant term (1s)
        
        A_size = self.n_samples + self.n_drift + 1
        A = np.zeros((A_size, A_size))
        
        # Fill variogram part
        A[:self.n_samples, :self.n_samples] = gamma_train
        
        # Fill drift part
        A[:self.n_samples, self.n_samples] = 1  # constant drift
        A[self.n_samples, :self.n_samples] = 1
        
        for i in range(self.n_drift):
            A[:self.n_samples, self.n_samples + 1 + i] = self.drift_train[:, i]
            A[self.n_samples + 1 + i, :self.n_samples] = self.drift_train[:, i]
        
        # Build right-hand side b for each prediction point
        b = np.zeros((n_pred, A_size))
        
        dist_pred = cdist(X_pred, self.X_train)
        gamma_pred = self._variogram(dist_pred)
        
        b[:, :self.n_samples] = gamma_pred
        b[:, self.n_samples] = 1  # constant drift
        
        # Handle drift prediction values
        drift_pred_array = np.atleast_1d(drift_pred)
        if drift_pred_array.ndim == 1:
            drift_pred_array = drift_pred_array.reshape(-1, 1)
        
        if drift_pred_array.shape[0] != n_pred:
            drift_pred_array = drift_pred_array.T
        
        for i in range(self.n_drift):
            b[:, self.n_samples + 1 + i] = drift_pred_array[:, i]
        
        eps = 1e-6
        A[:self.n_samples, :self.n_samples] += eps * np.eye(self.n_samples)

        return A, b
    
    def predict(self, X_pred: np.ndarray, drift_pred: np.ndarray, 
                return_std: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Predict values at new locations using KED.
        
        Parameters
        ----------
        X_pred : np.ndarray, shape (n_pred, n_features)
            Coordinates where predictions are needed
        drift_pred : np.ndarray, shape (n_pred, n_drift_vars)
            External drift values at prediction points
        return_std : bool, default=False
            Whether to return kriging standard deviation
            
        Returns
        -------
        y_pred : np.ndarray, shape (n_pred,)
            Predicted values
        sigma : np.ndarray, shape (n_pred,), optional
            Kriging standard deviation (if return_std=True)
        """
        if self.X_train is None:
            raise ValueError("Model must be fitted before prediction")
        
        X_pred = np.atleast_2d(X_pred)
        
        # Build kriging system
        A, b = self._build_kriging_matrix(X_pred, drift_pred)
        
        # Solve kriging system for weights
        weights = _solve_kriging(A, b)
        
        # Calculate predictions
        y_pred = weights[:, :self.n_samples] @ self.y_train
        
        if return_std:
            # Calculate kriging variance
            sigma_squared = np.sum(weights * b, axis=1)
            sigma = np.sqrt(np.maximum(sigma_squared, 0))  # Ensure non-negative
            return y_pred, sigma
        
        return y_pred, None


# Example usage and demonstration
def example_1d():
    """Simple 1D example with synthetic data"""
    print("=" * 60)
    print("Example 1: 1D Kriging with External Drift")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    n_train = 20
    
    # Training points
    X_train = np.linspace(0, 100, n_train).reshape(-1, 1)
    
    # External drift (e.g., elevation, temperature)
    drift_train = 0.5 * X_train.ravel() + 10 * np.sin(X_train.ravel() / 20)
    
    # Primary variable (correlated with drift + spatial variation)
    y_train = (2 * drift_train + 
               5 * np.sin(X_train.ravel() / 10) + 
               np.random.normal(0, 2, n_train))
    
    # Prediction points
    X_pred = np.linspace(0, 100, 200).reshape(-1, 1)
    drift_pred = 0.5 * X_pred.ravel() + 10 * np.sin(X_pred.ravel() / 20)
    
    # Fit KED model
    ked = KrigingExternalDrift(
        variogram_model='spherical',
        variogram_params={'sill': 50.0, 'range': 30.0, 'nugget': 2.0}
    )
    ked.fit(X_train, y_train, drift_train)
    
    # Predict
    y_pred, sigma = ked.predict(X_pred, drift_pred, return_std=True)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, y_train, c='red', s=50, label='Training data', zorder=3)
    plt.plot(X_pred, y_pred, 'b-', label='KED prediction', linewidth=2)
    plt.fill_between(X_pred.ravel(), 
                     y_pred - 2*sigma, 
                     y_pred + 2*sigma, 
                     alpha=0.3, label='±2σ confidence')
    plt.xlabel('Location')
    plt.ylabel('Value')
    plt.title('Kriging with External Drift - Predictions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.scatter(X_train, drift_train, c='green', s=50, label='Training drift', zorder=3)
    plt.plot(X_pred, drift_pred, 'g-', label='Prediction drift', linewidth=2)
    plt.xlabel('Location')
    plt.ylabel('Drift value')
    plt.title('External Drift Variable')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/ked_1d_example.png', dpi=150, bbox_inches='tight')
    print("✓ 1D example plot saved")
    
    return ked, y_pred, sigma


def example_2d():
    """2D spatial example"""
    print("\n" + "=" * 60)
    print("Example 2: 2D Kriging with External Drift")
    print("=" * 60)
    
    # Generate 2D synthetic data
    np.random.seed(42)
    n_train = 50
    
    # Random training locations
    X_train = np.random.uniform(0, 100, (n_train, 2))
    
    # External drift (e.g., elevation)
    drift_train = (0.3 * X_train[:, 0] + 
                   0.2 * X_train[:, 1] + 
                   10 * np.sin(X_train[:, 0] / 20))
    
    # Primary variable (e.g., soil moisture, temperature)
    y_train = (1.5 * drift_train + 
               10 * np.sin(X_train[:, 0] / 15) * np.cos(X_train[:, 1] / 15) +
               np.random.normal(0, 3, n_train))
    
    # Create prediction grid
    x_grid = np.linspace(0, 100, 50)
    y_grid = np.linspace(0, 100, 50)
    X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
    X_pred = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    
    drift_pred = (0.3 * X_pred[:, 0] + 
                  0.2 * X_pred[:, 1] + 
                  10 * np.sin(X_pred[:, 0] / 20))
    
    # Fit KED model
    ked = KrigingExternalDrift(
        variogram_model='exponential',
        variogram_params={'sill': 100.0, 'range': 40.0, 'nugget': 5.0}
    )
    ked.fit(X_train, y_train, drift_train)
    
    # Predict
    y_pred, sigma = ked.predict(X_pred, drift_pred, return_std=True)
    
    # Reshape for plotting
    Z_pred = y_pred.reshape(X_grid.shape)
    Z_sigma = sigma.reshape(X_grid.shape)
    
    # Plot results
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Prediction map
    im1 = axes[0].contourf(X_grid, Y_grid, Z_pred, levels=20, cmap='viridis')
    axes[0].scatter(X_train[:, 0], X_train[:, 1], c=y_train, 
                   s=80, edgecolors='white', linewidth=1.5, cmap='viridis')
    axes[0].set_xlabel('X coordinate')
    axes[0].set_ylabel('Y coordinate')
    axes[0].set_title('KED Predictions')
    plt.colorbar(im1, ax=axes[0], label='Predicted value')
    
    # Uncertainty map
    im2 = axes[1].contourf(X_grid, Y_grid, Z_sigma, levels=20, cmap='Reds')
    axes[1].scatter(X_train[:, 0], X_train[:, 1], c='blue', s=50, alpha=0.6)
    axes[1].set_xlabel('X coordinate')
    axes[1].set_ylabel('Y coordinate')
    axes[1].set_title('Prediction Uncertainty (σ)')
    plt.colorbar(im2, ax=axes[1], label='Standard deviation')
    
    # Drift map
    Z_drift = drift_pred.reshape(X_grid.shape)
    im3 = axes[2].contourf(X_grid, Y_grid, Z_drift, levels=20, cmap='terrain')
    axes[2].scatter(X_train[:, 0], X_train[:, 1], c='red', s=50, alpha=0.6)
    axes[2].set_xlabel('X coordinate')
    axes[2].set_ylabel('Y coordinate')
    axes[2].set_title('External Drift Variable')
    plt.colorbar(im3, ax=axes[2], label='Drift value')
    
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/ked_2d_example.png', dpi=150, bbox_inches='tight')
    print("✓ 2D example plot saved")
    
    return ked, Z_pred, Z_sigma


def _solve_kriging(A, b):
    """
    Solve kriging system with robust fallbacks.

    Parameters
    ----------
    A : ndarray, shape (M, M)
        Kriging matrix
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
            eps = 1e-6
            A_reg = A + eps * np.eye(A.shape[0])
            W = solve(A_reg, B)

    return W.T  # (n_pred, M)


if __name__ == "__main__":
    print("\nKriging with External Drift - Implementation Demo\n")
    
    # Run examples
    ked_1d, pred_1d, std_1d = example_1d()
    ked_2d, pred_2d, std_2d = example_2d()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("Plots saved to outputs directory")
    print("=" * 60)


