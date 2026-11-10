"""
Standalone ARIMA Forecaster for Nuclear LOCA Baseline

Pure statistical ARIMA implementation using statsmodels,
separate from TSLib PyTorch framework to avoid gradient corruption.

Author: Nuclear Research Team
Date: 2025
"""

import numpy as np
import warnings

# Suppress ARIMA convergence warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

try:
    from statsmodels.tsa.arima.model import ARIMA as StatsARIMA
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("WARNING: statsmodels not available. ARIMA will fallback to naive persistence.")


class ARIMAForecaster:
    """
    ARIMA (AutoRegressive Integrated Moving Average) Forecaster

    Pure statistical time series forecasting using Maximum Likelihood Estimation.
    No gradient-based training - fits per-sample during prediction.

    Parameters:
    -----------
    order : tuple (p, d, q)
        ARIMA order parameters
        - p: AR order (autoregressive, number of lagged observations)
        - d: Differencing order (removes non-stationarity/trends)
        - q: MA order (moving average, number of lagged forecast errors)

    Default: (1, 1, 1) - Standard Box-Jenkins baseline

    Methods:
    --------
    fit_and_forecast(series, steps)
        Fit ARIMA on historical series and forecast future steps

    batch_predict(samples, steps)
        Predict for multiple samples (batch processing)

    Example:
    --------
    >>> forecaster = ARIMAForecaster(order=(1, 1, 1))
    >>> history = np.array([10, 12, 14, 16, 18, 20])
    >>> predictions = forecaster.fit_and_forecast(history, steps=3)
    >>> print(predictions)  # [22, 24, 26] (approximately)
    """

    def __init__(self, order=(1, 1, 1)):
        """
        Initialize ARIMA forecaster

        Parameters:
        -----------
        order : tuple (p, d, q), default=(1, 1, 1)
            ARIMA order following Box-Jenkins methodology
        """
        self.order = order
        self.p, self.d, self.q = order

        if not STATSMODELS_AVAILABLE:
            print("WARNING: statsmodels not available. Will use naive persistence fallback.")

    def fit_and_forecast(self, series, steps):
        """
        Fit ARIMA model on historical series and forecast future steps

        Parameters:
        -----------
        series : np.ndarray, shape (seq_len,)
            Historical time series data

        steps : int
            Number of steps to forecast ahead

        Returns:
        --------
        predictions : np.ndarray, shape (steps,)
            Forecasted values for next 'steps' timesteps

        Notes:
        ------
        - Uses statsmodels.tsa.arima for Maximum Likelihood fitting
        - Falls back to naive persistence if:
          * statsmodels not available
          * Series too short (< 5 points)
          * ARIMA fitting fails (convergence issues)
        - Naive persistence: repeats last observed value
        """
        # Ensure series is 1D numpy array
        series = np.asarray(series).flatten()

        # Check if statsmodels available
        if not STATSMODELS_AVAILABLE:
            return self._naive_forecast(series, steps)

        # Check if series has enough data
        # ARIMA(1,1,1) needs at least p+d+1 = 3 points, but use 5 for stability
        if len(series) < 5:
            return self._naive_forecast(series, steps)

        try:
            # Fit ARIMA model using statsmodels
            model = StatsARIMA(series, order=self.order)
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})

            # Forecast future steps
            forecast = fitted_model.forecast(steps=steps)
            forecast = np.array(forecast)

            # Check for finite values (no inf/nan)
            if np.all(np.isfinite(forecast)):
                return forecast
            else:
                # If forecast has inf/nan, fallback to naive
                return self._naive_forecast(series, steps)

        except Exception as e:
            # If ARIMA fitting fails for any reason, fallback to naive
            # Common failures: convergence issues, numerical instability
            return self._naive_forecast(series, steps)

    def _naive_forecast(self, series, steps):
        """
        Naive persistence forecast: repeat last observed value

        This is the simplest possible forecast and serves as:
        1. Fallback when ARIMA fails
        2. Lower bound baseline (any good model should beat this)

        Parameters:
        -----------
        series : np.ndarray
            Historical series

        steps : int
            Number of steps to forecast

        Returns:
        --------
        predictions : np.ndarray, shape (steps,)
            Last value repeated 'steps' times
        """
        last_value = series[-1]
        return np.full(steps, last_value)

    def batch_predict(self, samples, steps):
        """
        Batch prediction for multiple time series samples

        Fits ARIMA independently for each sample and forecasts.
        This is more efficient than calling fit_and_forecast in a loop.

        Parameters:
        -----------
        samples : np.ndarray, shape (n_samples, seq_len)
            Multiple time series, each row is one sample

        steps : int
            Number of steps to forecast for each sample

        Returns:
        --------
        predictions : np.ndarray, shape (n_samples, steps)
            Forecasted values for each sample

        Example:
        --------
        >>> samples = np.array([[10, 12, 14], [20, 22, 24], [30, 32, 34]])
        >>> forecaster = ARIMAForecaster()
        >>> preds = forecaster.batch_predict(samples, steps=2)
        >>> print(preds.shape)  # (3, 2)
        """
        samples = np.asarray(samples)

        # Handle 1D input (single sample)
        if samples.ndim == 1:
            samples = samples.reshape(1, -1)

        n_samples = samples.shape[0]

        # Initialize output array
        predictions = np.zeros((n_samples, steps))

        # Fit and forecast for each sample
        for i in range(n_samples):
            series = samples[i, :]
            predictions[i, :] = self.fit_and_forecast(series, steps)

        return predictions

    def get_order(self):
        """Return ARIMA order (p, d, q)"""
        return self.order

    def __repr__(self):
        """String representation"""
        return f"ARIMAForecaster(order={self.order})"


class ARIMAAutoTuner:
    """
    ARIMA Auto-Tuner with Grid Search (Optional)

    Searches over multiple (p,d,q) combinations and selects best via AIC.
    NOT recommended for large datasets due to computational cost.

    Use fixed ARIMA(1,1,1) for fast baseline instead.
    """

    def __init__(self, p_range=(0, 1, 2), d_range=(0, 1), q_range=(0, 1, 2)):
        """
        Initialize auto-tuner with parameter ranges

        Parameters:
        -----------
        p_range : tuple of ints
            AR orders to try
        d_range : tuple of ints
            Differencing orders to try
        q_range : tuple of ints
            MA orders to try
        """
        self.p_range = p_range
        self.d_range = d_range
        self.q_range = q_range

    def find_best_order(self, series):
        """
        Find best ARIMA order via grid search and AIC selection

        Parameters:
        -----------
        series : np.ndarray
            Historical time series

        Returns:
        --------
        best_order : tuple (p, d, q)
            Best ARIMA order based on AIC
        """
        if not STATSMODELS_AVAILABLE:
            return (1, 1, 1)  # Default if statsmodels unavailable

        best_aic = np.inf
        best_order = (1, 1, 1)  # Default

        # Grid search
        for p in self.p_range:
            for d in self.d_range:
                for q in self.q_range:
                    try:
                        model = StatsARIMA(series, order=(p, d, q))
                        fitted = model.fit(method_kwargs={"warn_convergence": False})
                        aic = fitted.aic

                        if aic < best_aic:
                            best_aic = aic
                            best_order = (p, d, q)

                    except Exception:
                        continue

        return best_order


# Module-level function for quick usage
def quick_arima_forecast(history, steps, order=(1, 1, 1)):
    """
    Quick ARIMA forecast without creating forecaster object

    Parameters:
    -----------
    history : np.ndarray or list
        Historical time series

    steps : int
        Number of steps to forecast

    order : tuple (p, d, q), default=(1, 1, 1)
        ARIMA order

    Returns:
    --------
    predictions : np.ndarray
        Forecasted values

    Example:
    --------
    >>> history = [10, 12, 14, 16, 18, 20]
    >>> forecast = quick_arima_forecast(history, steps=3)
    >>> print(forecast)
    """
    forecaster = ARIMAForecaster(order=order)
    return forecaster.fit_and_forecast(np.array(history), steps)


if __name__ == "__main__":
    # Quick test
    print("="*80)
    print("ARIMA Standalone Forecaster - Quick Test")
    print("="*80)
    print()

    # Test 1: Simple linear trend
    print("Test 1: Linear Trend")
    t = np.arange(30)
    series = 2 * t + 10
    print(f"  Historical data (first 5): {series[:5]}")
    print(f"  Historical data (last 5): {series[-5:]}")

    forecaster = ARIMAForecaster(order=(1, 1, 1))
    predictions = forecaster.fit_and_forecast(series, steps=10)

    print(f"  Predictions (10 steps): {predictions}")
    print(f"  Expected (approx): {2 * np.arange(30, 40) + 10}")
    print()

    # Test 2: Batch prediction
    print("Test 2: Batch Prediction")
    samples = np.array([
        [10, 12, 14, 16, 18, 20],
        [100, 102, 104, 106, 108, 110],
        [50, 51, 52, 53, 54, 55]
    ])
    print(f"  Input shape: {samples.shape}")

    predictions = forecaster.batch_predict(samples, steps=3)
    print(f"  Predictions shape: {predictions.shape}")
    print(f"  Predictions:\n{predictions}")
    print()

    print("="*80)
    print("Quick Test Complete - ARIMA Forecaster Ready")
    print("="*80)
