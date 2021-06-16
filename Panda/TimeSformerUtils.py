import torch
import torch.nn as nn
from timesformer.models.vit import TimeSformer

model_urls = {
    'timesfomer-standard' : '/content/drive/MyDrive/anomaly_detection/timesformer_models/TimeSformer_divST_8x32_224_K400.pyth',
    'timesformer-HR': '/content/drive/MyDrive/anomaly_detection/timesformer_models/TimeSformer_divST_16_448_SSv2.pyth'
}

class TimeSformerWrapper(nn.Module):

    def __init__(self, mode='standard'):
        super(TimeSformerWrapper, self).__init__()
        if mode == 'timesformer-HR':
            self.timesformer_model = TimeSformer(img_size=448, num_classes=400, num_frames=16, attention_type='divided_space_time',
                                 pretrained_model=model_urls['timesformer-HR'])
        else:
            self.timesformer_model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                                 pretrained_model=model_urls['timesfomer-standard'])

        self.vision_transformer_model = self.timesformer_model.model

    def forward(self, x):
        features = self.vision_transformer_model.forward_features(x)
        x = self.vision_transformer_model.head(features)
        return x, features