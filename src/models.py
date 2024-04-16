from torch import nn
from torch.nn import init
from modules import NimbusLayer, NimbusLinear

class NimbusModel(nn.Module):
    def __init__(self) -> None:
        super(NimbusModel, self).__init__()
        self.nimbus_layers = []

    def get_nimbus_layers(self)->list[NimbusLayer]:
        if len(self.nimbus_layers) == 0:
            for module in self.modules():
                if isinstance(module, NimbusLayer):
                    self.nimbus_layers.append(module)
        return self.nimbus_layers


class Perceptron2(NimbusModel):
    def __init__(self, in_features, out_features, hidden_size1=32, hidden_size2=32):
        super(Perceptron2, self).__init__()
        self.bn0 = nn.BatchNorm1d(in_features)
        self.linear1 = NimbusLinear(in_features, hidden_size1)
        self.bn1 = nn.BatchNorm1d(hidden_size1)
        self.activation1 = nn.ReLU()
        self.linear2 = NimbusLinear(hidden_size1, hidden_size2)
        self.bn2 = nn.BatchNorm1d(hidden_size2)
        self.activation2 = nn.ReLU()
        self.linear3 = NimbusLinear(hidden_size2, out_features)

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
    
    # def get_nimbus_layers(self):
        # return [self.linear1, self.linear2, self.linear3]
