from torch import nn
from torch.nn import init
from modules import NimbusLinear


class Perceptron2(nn.Module):
    def __init__(self, in_features, out_features):
        super(Perceptron2, self).__init__()
        self.bn0 = nn.BatchNorm1d(in_features)
        self.linear1 = NimbusLinear(in_features, out_features)
        self.bn1 = nn.BatchNorm1d(out_features)
        self.activation1 = nn.ReLU()
        self.linear2 = NimbusLinear(out_features, out_features)
        self.bn2 = nn.BatchNorm1d(out_features)
        self.activation2 = nn.ReLU()
        self.linear3 = NimbusLinear(out_features, out_features)

    def forward(self, x):
        x = self.bn0(x)
        x = self.linear1(x)
        x = self.bn1(x)
        x = self.activation1(x)
        x = self.linear2(x)
        x = self.bn2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
