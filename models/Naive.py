import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Naive Persistence Baseline for Time Series Forecasting

    Architecture:
    - Repeats the last observed value for all future timesteps
    - Simplest possible forecasting baseline
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

        # Optional: Linear projection for feature dimension adjustment
        # For MS mode: enc_in features -> 1 output
        if self.c_out < self.enc_in:
            self.projection = nn.Linear(self.enc_in, self.c_out)
        else:
            self.projection = None

    def forecast(self, x_enc):
        """
        Forecasting task using naive persistence (last value repeated)

        Args:
            x_enc: [Batch, seq_len, enc_in] - Input sequence

        Returns:
            [Batch, pred_len, c_out] - Predictions (last value repeated)
        """
        batch_size = x_enc.size(0)

        # Use the last observed value
        last_value = x_enc[:, -1:, :]  # [B, 1, enc_in]

        # Project to output dimension if needed (MS mode)
        if self.projection is not None:
            last_value = self.projection(last_value)  # [B, 1, c_out]

        # Repeat the last value for all prediction steps
        # Naive assumption: future = present (no change)
        output = last_value.repeat(1, self.pred_len, 1)  # [B, pred_len, c_out]

        return output

    def imputation(self, x_enc):
        """
        Imputation task - forward-fill missing values with last observed

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        batch_size = x_enc.size(0)

        # For imputation, forward-fill: each timestep repeats previous value
        output = torch.zeros(batch_size, self.seq_len, self.c_out, device=x_enc.device)

        for t in range(self.seq_len):
            # Use the value at current timestep
            current_value = x_enc[:, t:t+1, :]  # [B, 1, enc_in]

            # Project if needed
            if self.projection is not None:
                current_value = self.projection(current_value)  # [B, 1, c_out]

            output[:, t:t+1, :] = current_value

        return output  # [B, seq_len, c_out]

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task - use persistence as expected value

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        # For anomaly detection, we could use previous value as expected
        # For simplicity, use same as imputation
        return self.imputation(x_enc)

    def classification(self, x_enc):
        """
        Classification task - use last value as feature

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, num_class]
        """
        # Not typically used for classification, but provide basic implementation
        # Use last value as feature
        last_value = x_enc[:, -1, :]  # [B, enc_in]

        # Simple linear classifier (would need to add this in __init__ for real use)
        # For now, return zeros as placeholder
        raise NotImplementedError("Naive not designed for classification tasks")

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
