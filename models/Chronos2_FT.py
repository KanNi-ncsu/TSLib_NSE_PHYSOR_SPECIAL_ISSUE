import torch
from torch import nn
from chronos import BaseChronosPipeline


class Model(nn.Module):
    """
    TSLib wrapper for fine-tuned Chronos-2 model.

    Loads a fine-tuned Chronos2Pipeline from a local checkpoint path
    specified via configs.finetuned_model_path. Otherwise identical to
    the zero-shot Chronos2 wrapper.
    """

    def __init__(self, configs):
        super().__init__()
        # Load fine-tuned model from local checkpoint
        model_path = getattr(configs, "finetuned_model_path", None)
        if model_path is None:
            raise ValueError(
                "Chronos2_FT requires --finetuned_model_path pointing to "
                "a fine-tuned Chronos-2 checkpoint directory"
            )
        print(f"Loading fine-tuned Chronos-2 from: {model_path}")
        self.model = BaseChronosPipeline.from_pretrained(model_path, device_map="cuda")
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Instance normalization (same as zero-shot Chronos2)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc.sub(means)
        stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc = x_enc.div(stdev)

        B, L, C = x_enc.shape
        x_enc = x_enc.permute(0, 2, 1)
        quantiles, dec_out = self.model.predict_quantiles(
            x_enc.cpu().numpy(),
            prediction_length=self.pred_len,
            quantile_levels=[0.1, 0.5, 0.9],
        )
        dec_out = torch.stack(dec_out, dim=0).to(x_enc.device)
        dec_out = dec_out.permute(0, 2, 1)

        # Reverse normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "zero_shot_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
