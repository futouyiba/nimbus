import torch
from defines import *
import arg_settings
import torch.nn as nn

from models import NimbusModel
from modules import NimbusLayer, NimbusLinear

# 训练过程分为几个阶段：
# - 初训。首先使用普通的矩阵乘法、神经网络、反向传播，来获得一个训练基准。这个阶段主要达成几个目的：
# -- 给DM的LUT优化提供input
# -- 给MADDNESS的分桶提供参数矩阵（可看成MADDNESS论文中的B矩阵）。在神经网络的背景下，参数矩阵通常指的是weight矩阵
# - DM化（Differentiable MADDNESSize）

#   逐层DM化，并锁住其他层进行单层训练。这个过程中，Nimbus Layer在几种状态中来回切换，以达到训练目的。这是一个对于Nimbus Layer的enumerate过程，对其中的第l层，基本操作为：
#   - 将l-1层切换状态（l==1的处理省略），使用高速的MADDNESS算法进行推理，输出数据给第l层作为输入使用
#   - 将l层切换状态，使用selection matrix、tree des matrix来进行可微的MADDNESS计算并进行梯度降落，以使得
#   - 后面的L-l层都锁住，使用普通MatMul方法计算、传递梯度，但不做梯度降落
# - 量化（前期跳过）。使用8位定点数代替浮点数。
# - 微调。这个阶段里，每个Nimbus Layer都植入DM状态，进行较为缓慢的DM梯度降落。这一步的成本由于前几步而降低，尤其是在当前主流的深度学习现状下，典型的层数>8，<100(TODO)，这时受益显著。
# - 终优化。使用变换后的LUT取代BN层、激活层等，达到进一步的速度提升。

# TODO put these into function body instead of global variables
LastDMizedLayer:NimbusLayer = None
CurDMizedLayer:NimbusLayer = None

def SaveInputCache(model: nn.Module):
    pass

def train_epochs_from_setting(model: nn.Module):
    pass

def evaluate_model(model: nn.Module):
    pass

# 把模型中某一层替换为可微的MADDNESS层，锁定其他层不进行更新。这个过程supposedly被逐个使用，比如一个3层网络，在第一层DM化（differentiable MADDNESS化）之后，
# 第一层以MADDNESS only的方式进行推导，且不接受梯度更新，第二层以DM的方式进行推导，且接受梯度更新，第三层以普通mm的方式进行推导，不接受梯度更新。
# 这样保持了一定的速度，又能逐层的进行DM化，完成量化后重训练。
# Substitutes a layer in the model with a differentiable MADDNESS layer, and locks other layers for gradient update.
# This process is supposed to be used iteratively, for example, a 3-layer network, after the first layer is DM-ized (differentiable MADDNESS-ized),
# the first layer is derived in MADDNESS only mode, and does not accept gradient updates, the second layer is derived in DM mode, and accepts gradient updates,
# the third layer is derived in normal mm mode, and does not accept gradient updates.
# This way, a certain speed is maintained, and the DM-ization is done layer by layer, and retraining is done after quantization.
def DM_next_layer(model: NimbusModel)->bool:
    nimbusLayers = model.get_nimbus_layers()
    
    if LastDMizedLayer is None:
        CurDMizedLayer = nimbusLayers[0]
    else:
        LastDMizedLayer = CurDMizedLayer
        if nimbusLayers.index(LastDMizedLayer) == len(nimbusLayers) - 1:
            return False
        CurDMizedLayer = nimbusLayers[nimbusLayers.index(LastDMizedLayer) + 1]

    LastDMizedLayer.set_state(NIMBUS_STATE_MADDNESS_ONLY)
    CurDMizedLayer.set_state(NIMBUS_STATE_DM_BACKPROP)


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
        train_epochs_from_setting(model)
        SaveInputCache(model)
        pass
    if STEP_DIFFERENTIABLE_MADDNESS_LAYERS & arg_settings.TrainProcessesInChain:
        # replace each layer with differentiable MADDNESS layer, and retrain model iteratively
        for l in nimbusLayers:
            DM_next_layer(model)
            train_epochs_from_setting(model)
        pass
    if STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS & arg_settings.TrainProcessesInChain:
        # fine tune the model
        for l in nimbusLayers:
            l.set_state(NIMBUS_STATE_DM_BACKPROP)
        train_epochs_from_setting(model)
        pass
    if STEP_EVALUATE_MADDNESS_ONLY & arg_settings.TrainProcessesInChain:
        # evaluate maddness-only model
        pass
    #endregion