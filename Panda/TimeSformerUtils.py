import torch
import torch.nn as nn
from timesformer.models.vit import TimeSformer

model_urls = {
    'timesformer-HR': '/content/drive/MyDrive/anomaly_detection/timesformer_models/TimeSformer_divST_16_448_SSv2.pyth'
}

class TimeSformerWrapper(nn.Module):

    def __init__(self, model_path=model_urls['timesformer-HR']):
        super(TimeSformerWrapper, self).__init__()
        self.timesformer_model = TimeSformer(img_size=448, num_classes=400, num_frames=16, attention_type='divided_space_time',
                                 pretrained_model=model_path)

        self.vision_transformer_model = self.timesformer_model.model

    def forward(self, x):
        features = self.vision_transformer_model.forward_features(x)
        x = self.vision_transformer_model.head(x)
        return x, features