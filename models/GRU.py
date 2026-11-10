import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """
    Vanilla GRU (Gated Recurrent Unit) for Time Series Forecasting

    Architecture:
    - Multi-layer GRU encoder
    - Linear projection to prediction horizon
    - Supports multivariate input/output
    - Handles normalization for better training stability

    Compatible with TSLib framework for nuclear LOCA forecasting

    GRU is similar to LSTM but more efficient (no cell state)
    """

    def __init__(self, configs):
        super(Model, self).__init__()

        # Get parameters from configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len if self.task_name in ['long_term_forecast', 'short_term_forecast'] else configs.seq_len

        self.enc_in = configs.enc_in
        self.c_out = configs.c_out
        self.d_model = configs.d_model
        self.dropout = configs.dropout
        self.num_layers = configs.e_layers  # Use encoder layers config

        # GRU Encoder
        # input_size: number of input features (e.g., 74 for LOCA)
        # hidden_size: hidden state dimension (d_model, e.g., 128)
        # num_layers: number of stacked GRU layers (e.g., 2)
        self.gru = nn.GRU(
            input_size=self.enc_in,
            hidden_size=self.d_model,
            num_layers=self.num_layers,
            dropout=self.dropout if self.num_layers > 1 else 0,  # Dropout only if multiple layers
            batch_first=True,
            bidirectional=False
        )

        # Projection layers for different tasks
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            # For forecasting: project from hidden state to prediction horizon
            self.projection = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.pred_len * self.c_out)
            )
        elif self.task_name in ['imputation', 'anomaly_detection']:
            # For imputation/anomaly detection: reconstruct input sequence
            self.projection = nn.Sequential(
                nn.Dropout(self.dropout),
                nn.Linear(self.d_model, self.c_out)
            )
        elif self.task_name == 'classification':
            # For classification: global pooling + classifier
            self.act = F.gelu
            self.dropout_layer = nn.Dropout(self.dropout)
            self.projection = nn.Linear(self.d_model * self.seq_len, configs.num_class)

    def forecast(self, x_enc):
        """
        Forecasting task

        Args:
            x_enc: [Batch, seq_len, enc_in] - Input sequence

        Returns:
            [Batch, pred_len, c_out] - Predictions
        """
        batch_size = x_enc.size(0)

        # Normalization: subtract last value for stability
        # This helps GRU learn changes rather than absolute values
        seq_last = x_enc[:, -1:, :].detach()  # [B, 1, enc_in]
        x_enc = x_enc - seq_last  # [B, seq_len, enc_in]

        # GRU encoding
        # gru_out: [B, seq_len, d_model]
        # h_n: [num_layers, B, d_model] - only hidden state, no cell state
        gru_out, h_n = self.gru(x_enc)

        # Use the last hidden state from the final layer
        # h_n[-1]: [B, d_model]
        final_hidden = h_n[-1]  # [B, d_model]

        # Project to prediction horizon
        # output: [B, pred_len * c_out]
        output = self.projection(final_hidden)

        # Reshape to [B, pred_len, c_out]
        output = output.view(batch_size, self.pred_len, self.c_out)

        # Denormalization: add back the last value
        # For MS mode, seq_last has enc_in features but we only output c_out (1)
        # So we take the target feature (last column in MS mode)
        if self.c_out < self.enc_in:
            # MS mode: only use the target feature for denormalization
            seq_last = seq_last[:, :, -self.c_out:]  # [B, 1, c_out]

        output = output + seq_last

        return output  # [B, pred_len, c_out]

    def imputation(self, x_enc):
        """
        Imputation task - reconstruct missing values

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        batch_size = x_enc.size(0)

        # Normalization
        seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - seq_mean

        # GRU encoding - get output at each timestep
        gru_out, _ = self.gru(x_enc)  # [B, seq_len, d_model]

        # Project each timestep
        output = self.projection(gru_out)  # [B, seq_len, c_out]

        # Denormalization
        if self.c_out < self.enc_in:
            seq_mean = seq_mean[:, :, -self.c_out:]
        output = output + seq_mean

        return output  # [B, seq_len, c_out]

    def anomaly_detection(self, x_enc):
        """
        Anomaly detection task - reconstruct sequence to detect anomalies

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, seq_len, c_out]
        """
        # Same as imputation for GRU
        return self.imputation(x_enc)

    def classification(self, x_enc):
        """
        Classification task

        Args:
            x_enc: [Batch, seq_len, enc_in]

        Returns:
            [Batch, num_class]
        """
        # Normalization
        seq_mean = x_enc.mean(dim=1, keepdim=True).detach()
        x_enc = x_enc - seq_mean

        # GRU encoding
        gru_out, _ = self.gru(x_enc)  # [B, seq_len, d_model]

        # Global representation: flatten sequence
        output = self.act(gru_out)  # [B, seq_len, d_model]
        output = self.dropout_layer(output)
        output = output.reshape(output.shape[0], -1)  # [B, seq_len * d_model]

        # Classification
        output = self.projection(output)  # [B, num_class]

        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        """
        Forward pass - routes to appropriate task

        Args:
            x_enc: [Batch, seq_len, enc_in] - Encoder input
            x_mark_enc: [Batch, seq_len, mark_dim] - Encoder time features (not used)
            x_dec: [Batch, label_len+pred_len, dec_in] - Decoder input (not used for GRU)
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
