from pykrige import OrdinaryKriging
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
import numpy as np

from sklearn.linear_model import LinearRegression


class SpatioTemporalNNGP:
    def __init__(self, kernel, alpha=1e-10, phi=0.9, tau2=0.1):
        self.kernel = kernel
        self.alpha = alpha
        self.phi = phi
        self.tau2 = tau2
        
        # Initialize placeholders
        self.beta = None
        self.nngp_spatial_model = GaussianProcessRegressor(
            kernel=self.kernel, alpha=self.alpha,
        )

    def fit(self, X_flat, y_flat):
        lr = LinearRegression()
        lr.fit(X_flat, y_flat)
        self.beta = lr.coef_
        self.beta = self.beta.transpose(1, 0)
        self.intercept = lr.intercept_

    def fit_residual(self, X_0, y_0):
        # Compute residuals after removing mean structure (autoregressive part handled separately)
        residuals_partial = y_0 - (X_0 @ self.beta + self.intercept)

        # Fit the spatial NNGP model on residuals of the first time step
        self.nngp_spatial_model.fit(X_0[:, :2], residuals_partial)

        # Store residuals for autoregressive part
        self.residuals = residuals_partial

    def predict(self, X_0, y_0, X_pred, return_std=False):
        X_pred = X_pred.reshape(X_pred.shape[0], 1, X_pred.shape[1])
        n_pred_times, n_pred_locations, _ = X_pred.shape

        predictions = np.zeros((n_pred_times, n_pred_locations))
        std_devs = np.zeros_like(predictions)

        for t in range(n_pred_times):
            # self.fit_residual(X_0[t], y_0[t])

            mean_structure = X_pred[t] @ self.beta + self.intercept
            
            # Spatial prediction only
            # pred, std = self.nngp_spatial_model.predict(X_pred[t][:, :2], return_std=True)
            ok = OrdinaryKriging(
                X_0[t][:, 0:1], X_0[t][:, 1:2], y_0[t],
                variogram_model='linear',
            )
            pred, _ = ok.execute('points', [X_pred[t][:, 0:1]], [X_pred[t][:, 1:2]])
            predictions[t] = mean_structure + pred[0]
            # std_devs[t] = np.sqrt(std**2 + self.tau2)

            # if t == 0:
            #     # Spatial prediction only
            #     pred, std = self.nngp_spatial_model.predict(X_pred[t][:, :2], return_std=True)
            #     predictions[t] = mean_structure + pred
            #     std_devs[t] = np.sqrt(std**2 + self.tau2)
            # else:
            #     # Include autoregressive temporal dependency
            #     temporal_component = self.phi * (predictions[t-1] - (X_pred[t-1] @ self.beta + self.intercept))
            #     pred, std = self.nngp_spatial_model.predict(X_pred[t][:, :2], return_std=True)
            #     predictions[t] = mean_structure + temporal_component + pred
            #     std_devs[t] = np.sqrt(std**2 + self.tau2)

        # if return_std:
        #     return predictions, std_devs
        # else:
        return predictions

    def predict_legacy(self, X_0, y_0, X_pred, return_std=False):
        X_pred = X_pred.reshape(X_pred.shape[0], 1, X_pred.shape[1])
        n_pred_times, n_pred_locations, _ = X_pred.shape

        predictions = np.zeros((n_pred_times, n_pred_locations))
        std_devs = np.zeros_like(predictions)

        for t in range(n_pred_times):
            self.fit_residual(X_0[t], y_0[t])

            mean_structure = X_pred[t] @ self.beta + self.intercept
            
            # Spatial prediction only
            pred, std = self.nngp_spatial_model.predict(X_pred[t][:, :2], return_std=True)
            predictions[t] = mean_structure + pred
            std_devs[t] = np.sqrt(std**2 + self.tau2)

            # if t == 0:
            #     # Spatial prediction only
            #     pred, std = self.nngp_spatial_model.predict(X_pred[t][:, :2], return_std=True)
            #     predictions[t] = mean_structure + pred
            #     std_devs[t] = np.sqrt(std**2 + self.tau2)
            # else:
            #     # Include autoregressive temporal dependency
            #     temporal_component = self.phi * (predictions[t-1] - (X_pred[t-1] @ self.beta + self.intercept))
            #     pred, std = self.nngp_spatial_model.predict(X_pred[t][:, :2], return_std=True)
            #     predictions[t] = mean_structure + temporal_component + pred
            #     std_devs[t] = np.sqrt(std**2 + self.tau2)

        if return_std:
            return predictions, std_devs
        else:
            return predictions
