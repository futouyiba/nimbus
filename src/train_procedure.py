import torch
from defines import *
import arg_settings
import torch.nn as nn

from models import NimbusModel
from modules import NimbusLinear

def SaveInputCache(model: nn.Module):
    pass

def train_one_step(model: nn.Module):
    pass

def evaluate_model(model: nn.Module):
    pass

# 把模型中某一种替换为可微的MADDNESS层，锁定其他层不进行更新。这个过程supposedly被逐个使用，比如一个3层网络，在第一层DM化（differentiable MADDNESS化）之后，
# 第一层以MADDNESS only的方式进行推导，且不接受梯度更新，第二层以DM的方式进行推导，且接受梯度更新，第三层以普通mm的方式进行推导，不接受梯度更新。
# 这样保持了一定的速度，又能逐层的进行DM化，完成量化后重训练。
# Substitutes a layer in the model with a differentiable MADDNESS layer, and locks other layers for gradient update.
# This process is supposed to be used iteratively, for example, a 3-layer network, after the first layer is DM-ized (differentiable MADDNESS-ized),
# the first layer is derived in MADDNESS only mode, and does not accept gradient updates, the second layer is derived in DM mode, and accepts gradient updates,
# the third layer is derived in normal mm mode, and does not accept gradient updates.
# This way, a certain speed is maintained, and the DM-ization is done layer by layer, and retraining is done after quantization.
def DM_next_layer(model: NimbusModel):
    nimbusLayers = model.get_nimbus_layers()
    nextLayerProcessed = False
    for i in range(len(nimbusLayers)):
        layer = nimbusLayers[i]

        if nimbusLayers[i].state == NIMBUS_STATE_MATMUL:
            if not nextLayerProcessed:
                nimbusLayers[i].state = NIMBUS_STATE_MADDNESS_BACKPROP
            
            continue
        if nimbusLayers[i].state == NIMBUS_STATE_MADDNESS_BACKPROP:
            nimbusLayers[i].state = NIMBUS_STATE_MADDNESS_ONLY
            continue
        if nimbusLayers[i].state == NIMBUS_STATE_MADDNESS_ONLY:
            continue

def lock_all_nimbus_layers(model: NimbusModel):
    nimbusLayers = model.get_nimbus_layers()
    for i in range(len(nimbusLayers)):
        layer = nimbusLayers[i]
        # prevent gradient update
        for param in layer.parameters():
            param.requires_grad = False

def nimbus_layer_require_grad(layer: NimbusLinear):
    layer.lut.requires_grad = True
    layer.thresholds.requires_grad = True
    layer.bias.requires_grad = True
    layer.weight.requires_grad = True

def train_procedure():
    model = arg_settings.TrainModel  # Assign the value of arg_settings.TrainModel to the variable model
    model = model.to(arg_settings.Device)
    model.train()

    nimbusLayers = model.get_nimbus_layers()

    #region data

    #region training
    if STEP_LINEAR_ONLY & arg_settings.TrainProcessesInChain:
        # train the model
        train_one_step(model)
        SaveInputCache(model)
        pass
    if STEP_DIFFERENTIABLE_MADDNESS_LAYERS & arg_settings.TrainProcessesInChain:
        # replace each layer with differentiable MADDNESS layer, and retrain model iteratively
        for l in nimbusLayers:
            
            train_one_step(model)
        pass
    if STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS & arg_settings.TrainProcessesInChain:
        # fine tune the model
        pass
    if STEP_EVALUATE_MADDNESS_ONLY & arg_settings.TrainProcessesInChain:
        # evaluate maddness-only model
        pass
    #endregion