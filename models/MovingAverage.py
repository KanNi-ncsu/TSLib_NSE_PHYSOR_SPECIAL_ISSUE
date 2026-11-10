import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Moving Average Baseline for Time Series Forecasting

    Architecture:
    - Computes mean of last K timesteps (window_size)
    - Projects mean to prediction horizon
    - Simple statistical baseline model
    - No learnable parameters (except optional linear projection)

    Compatible with TSLib framework for nuclear LOCA forecasting
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Get parameters from configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if self.task_name in ['long_term_forecast', 'short_term_forecast'] else configs.seq_len

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out

        # Window size for moving average
        # Use full sequence length for maximum context
        self.window_size = self.seq_len

        # Optional: Linear projection for feature dimension adjustment
        # For MS mode: enc_in features -> 1 output
        if self.c_out < self.enc_in:
            self.projection = nn.Linear(self.enc_in, self.c_out)
        else:
            self.projection = None

    def forecast(self, x_enc):
        """
        Forecasting task using moving average

        Args:
            x_enc: [Batch, seq_len, enc_in] - Input sequence

        Returns:
            [Batch, pred_len, c_out] - Predictions (constant value repeated)
        """
        batch_size = x_enc.size(0)

        # Compute moving average over the last window_size timesteps
        # Use all available timesteps (seq_len) for more stable average
        window = x_enc[:, -self.window_size:, :]  # [B, window_size, enc_in]
        avg = window.mean(dim=1, keepdim=True)  # [B, 1, enc_in]

        # Project to output dimension if needed (MS mode)
        if self.projection is not None:
            avg = self.projection(avg)  # [B, 1, c_out]

        # Repeat the average value for all prediction steps
        # This is a naive approach: same value for all future timesteps
        output = avg.repeat(1, self.pred_len, 1)  # [B, pred_len, c_out]

        return output

    def imputation(self, x_enc):
        """
        Imputation task - fill missing values with moving average

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        batch_size = x_enc.size(0)

        # For imputation, compute moving average at each timestep
        # Use cumulative mean for simplicity
        output = torch.zeros(batch_size, self.seq_len, self.c_out, device=x_enc.device)

        for t in range(self.seq_len):
            # Average of all previous timesteps (including current)
            window = x_enc[:, :t+1, :]  # [B, t+1, enc_in]
            avg = window.mean(dim=1, keepdim=True)  # [B, 1, enc_in]

            # Project if needed
            if self.projection is not None:
                avg = self.projection(avg)  # [B, 1, c_out]

            output[:, t:t+1, :] = avg

        return output  # [B, seq_len, c_out]

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task - use moving average as expected value

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        # Same as imputation for MovingAverage
        return self.imputation(x_enc)

    def classification(self, x_enc):
        """
        Classification task - use overall average as feature

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, num_class]
        """
        # Not typically used for classification, but provide basic implementation
        # Average over entire sequence
        avg = x_enc.mean(dim=1)  # [B, enc_in]

        # Simple linear classifier (would need to add this in __init__ for real use)
        # For now, return zeros as placeholder
        # In practice, would need configs.num_class
        raise NotImplementedError("MovingAverage not designed for classification tasks")

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
