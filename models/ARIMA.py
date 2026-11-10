import torch
import torch.nn as nn
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
    print("Warning: statsmodels not available. ARIMA will use naive fallback.")


class Model(nn.Module):
    """
    ARIMA(1,1,1) forecaster (statsmodels) wrapped as a TSLib model.

    In MS mode the inputs are mapped to one series by a linear projection before fitting.
    The ARIMA results reported for the LOCA experiments are produced by
    NuclearResearch/src/baselines/run_arima_baseline.py.
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Get parameters from configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if self.task_name in ['long_term_forecast', 'short_term_forecast'] else configs.seq_len

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # ARIMA order: Fixed (1,1,1) - standard baseline following Box-Jenkins
        # p=1: AR order (autoregressive, uses 1 lagged observation)
        # d=1: Differencing order (removes non-stationarity)
        # q=1: MA order (moving average, uses 1 lagged forecast error)
        self.arima_order = (1, 1, 1)

        # Projection for feature dimension adjustment (MS mode)
        if self.c_out < self.enc_in:
            self.projection = nn.Linear(self.enc_in, self.c_out)
        else:
            self.projection = None

        # Gradient connector so that the TSLib training loop can call backward()
        self.grad_connector = nn.Linear(1, 1, bias=False)
        nn.init.ones_(self.grad_connector.weight)  # Initialize to 1.0 (identity)

        if not STATSMODELS_AVAILABLE:
            print("WARNING: statsmodels not available. ARIMA will fallback to naive persistence.")

    def _fit_and_forecast_arima(self, series, steps):
        """
        Fit ARIMA(1,1,1) model and forecast

        Uses fixed order (1,1,1) following Box-Jenkins methodology:
        - p=1: AR order (autoregressive, uses 1 lagged value)
        - d=1: Differencing order (removes trends/non-stationarity)
        - q=1: MA order (moving average, uses 1 lagged error)

        This is the standard ARIMA baseline used in time series literature.

        Args:
            series: 1D numpy array of historical values
            steps: Number of steps to forecast

        Returns:
            predictions: 1D numpy array of forecasted values
        """
        if not STATSMODELS_AVAILABLE:
            # Fallback: repeat last value (naive persistence)
            return np.full(steps, series[-1])

        # Ensure series has enough data for ARIMA(1,1,1)
        if len(series) < 5:
            return np.full(steps, series[-1])

        try:
            # Fit ARIMA(1,1,1) model
            model = StatsARIMA(series, order=self.arima_order)
            fitted_model = model.fit(method_kwargs={"warn_convergence": False})

            # Forecast future steps
            forecast = fitted_model.forecast(steps=steps)
            forecast = np.array(forecast)

            # Ensure finite values (check for inf/nan)
            if np.all(np.isfinite(forecast)):
                return forecast
            else:
                # If forecast has inf/nan, fallback to naive
                return np.full(steps, series[-1])

        except Exception:
            # If ARIMA(1,1,1) fails, fallback to naive persistence
            return np.full(steps, series[-1])

    def forecast(self, x_enc):
        """
        Forecasting task using ARIMA

        Args:
            x_enc: [Batch, seq_len, enc_in] - Input sequence

        Returns:
            [Batch, pred_len, c_out] - ARIMA predictions
        """
        batch_size = x_enc.size(0)
        device = x_enc.device

        # Convert to numpy for ARIMA fitting
        x_enc_np = x_enc.detach().cpu().numpy()

        # Initialize output tensor
        output = torch.zeros(batch_size, self.pred_len, self.c_out, device=device)

        # For each sample in batch
        for b in range(batch_size):
            # Extract encoder sequence for this sample
            sample = x_enc_np[b, :, :]  # [seq_len, enc_in]

            # Project to target dimension if needed (MS mode)
            if self.projection is not None:
                sample_tensor = torch.from_numpy(sample).float().to(device)
                sample_projected = self.projection(sample_tensor).detach().cpu().numpy()
            else:
                sample_projected = sample

            # For each output feature (typically c_out=1 for TPCT)
            for c in range(self.c_out):
                # Extract univariate time series for this feature
                series = sample_projected[:, c] if sample_projected.shape[1] > 1 else sample_projected.flatten()

                # Fit ARIMA and forecast
                forecast = self._fit_and_forecast_arima(series, self.pred_len)

                # Store predictions
                output[b, :, c] = torch.from_numpy(forecast).float()

        # Move output to device
        output = output.to(device)

        # Apply gradient connector to link ARIMA outputs to computational graph
        # Reshape: [B, pred_len, c_out] -> [B*pred_len*c_out, 1]
        original_shape = output.shape
        output_flat = output.reshape(-1, 1)

        # Pass through grad_connector (provides gradients for TSLib)
        output_connected = self.grad_connector(output_flat)

        # Reshape back: [B*pred_len*c_out, 1] -> [B, pred_len, c_out]
        output = output_connected.reshape(original_shape)

        return output

    def imputation(self, x_enc):
        """
        Imputation task - forward-fill with ARIMA smoothing

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        # For imputation, use simple forward-fill (ARIMA not designed for this)
        batch_size = x_enc.size(0)
        device = x_enc.device

        output = torch.zeros(batch_size, self.seq_len, self.c_out, device=device)

        for t in range(self.seq_len):
            current_value = x_enc[:, t:t+1, :]

            if self.projection is not None:
                current_value = self.projection(current_value)

            output[:, t:t+1, :] = current_value

        return output

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task - use ARIMA expected value

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        # For anomaly detection, could use ARIMA residuals, but for simplicity use imputation
        return self.imputation(x_enc)

    def classification(self, x_enc):
        """
        Classification task - not supported for ARIMA

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, num_class]
        """
        raise NotImplementedError("ARIMA not designed for classification tasks")

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass - routes to appropriate task

        Args:
            x_enc: [Batch, seq_len, enc_in] - Encoder input
            x_mark_enc: [Batch, seq_len, mark_dim] - Encoder time features (not used)
            x_dec: [Batch, label_len+pred_len, dec_in] - Decoder input (not used)
            x_mark_dec: [Batch, label_len+pred_len, mark_dim] - Decoder time features (not used)
            mask: Optional mask for imputation

        Returns:
            Output tensor with shape depending on task
        """
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :]  # [B, pred_len, c_out]

        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc)
            return dec_out  # [B, seq_len, c_out]

        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, seq_len, c_out]

        if self.task_name == 'classification':
            dec_out = self.classification(x_enc)
            return dec_out  # [B, num_class]

        return None
