import torch

class JointModel(torch.nn.Module):
    def __init__(self, feature_extractor, classifer):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifer

    def get_features(self, x):
        return self.feature_extractor(x)

    def forward(self, x):
        features = self.get_features(x)
        y = self.classifier(features)
        return y

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)