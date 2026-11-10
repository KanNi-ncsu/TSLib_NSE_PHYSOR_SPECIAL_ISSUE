import torch
from torch import nn
from chronos import BaseChronosPipeline


class Model(nn.Module):
    """
    TSLib wrapper for fine-tuned Chronos Bolt model.

    Loads a fine-tuned ChronosBoltPipeline from a local checkpoint path
    specified via configs.finetuned_model_path. Otherwise identical to
    the zero-shot Chronos wrapper (loops over channels, takes mean of
    sample dimension).
    """

    def __init__(self, configs):
        super().__init__()
        model_path = getattr(configs, "finetuned_model_path", None)
        if model_path is None:
            raise ValueError(
                "Chronos_FT requires --finetuned_model_path pointing to "
                "a fine-tuned Chronos Bolt checkpoint directory"
            )
        print(f"Loading fine-tuned Chronos Bolt from: {model_path}")
        self.model = BaseChronosPipeline.from_pretrained(
            model_path,
            device_map="cuda",
            torch_dtype=torch.bfloat16,
        )
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        outputs = []
        for i in range(x_enc.shape[-1]):
            output = self.model.predict(x_enc[..., i], prediction_length=self.pred_len)
            output = output.mean(dim=1)
            outputs.append(output)
        dec_out = torch.stack(outputs, dim=-1)
        return dec_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "zero_shot_forecast":
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out
        return None
