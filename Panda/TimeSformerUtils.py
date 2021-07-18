import torch
import torch.nn as nn
from timesformer.models.vit import TimeSformer

model_urls = {
    'timesfomer-standard' : '/vol/ep/mm/anomaly_detection/data/TimeSformer_divST_8x32_224_K400.pyth',
    'timesformer-HR': '/vol/ep/mm/anomaly_detection/data/TimeSformer_divST_16_448_SSv2.pyth'
}

class TimeSformerWrapper(nn.Module):

    def __init__(self, mode='standard'):
        super(TimeSformerWrapper, self).__init__()
        self.mode = mode
        if mode == 'hr':
            self.timesformer_model = TimeSformer(img_size=448, num_classes=400, num_frames=16, attention_type='divided_space_time',
                                 pretrained_model=model_urls['timesformer-HR'])
        else:
            self.timesformer_model = TimeSformer(img_size=224, num_classes=400, num_frames=8, attention_type='divided_space_time',
                                 pretrained_model=model_urls['timesfomer-standard'])

        self.vision_transformer_model = self.timesformer_model.model

    def forward(self, x):
        if self.mode == 'standard' and x.shape[1] == 16:
            bottom_frames = x[:,:8,:,:]
            top_frames = x[:,9:,:,:]
            bottom_features = self.vision_transformer_model.forward_features(bottom_frames)
            top_features = self.vision_transformer_model.forward_features(top_frames)
            features = torch.cat([bottom_features, top_features], dim=1)

            bottom_x = self.vision_transformer_model.head(bottom_features)
            top_x = self.vision_transformer_model.head(top_features)
            x = (bottom_x + top_x) / 2.0
        else:
            features = self.vision_transformer_model.forward_features(x)
            x = self.vision_transformer_model.head(features)
        return x, features
