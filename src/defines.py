import os
import models

#region path settings
CWD_PREFIX = "."

DATA_PATH_ROOT = "data"
LOGS_PATH_ROOT = "logs"

SAVES_PATH_ROOT = "saves"
INPUT_CACHE_PATH_ROOT = os.path.join(SAVES_PATH_ROOT, "input_cache")
MODEL_CHECKPOINT_PATH_ROOT = os.path.join(SAVES_PATH_ROOT, "model_checkpoints")
RUNS_PATH_ROOT = os.path.join(SAVES_PATH_ROOT, "runs")
#endregion


#region Consts:NimbusLayer
# used for Halut layer state flags
# 下面这些状态常量相当于枚举，用于标识Nimbus层的状态，以便在训练过程中进行不同的操作。
# 根据训练阶段和激活次序，Nimbus层会在不同的状态下进行不同的操作。Nimbus层不仅管理自己的操作，还会根据情况跳过一些层比如BN、ReLU等。

# 这个状态下，使用普通的矩阵乘法，慢但稳妥、精确、可反向传播。初始化模型时，所有层都是这个状态。
# 这个状态会用来进行正常的训练，并得到一个基准模型。另外，存input也用这个状态进行。
NIMBUS_STATE_MATMUL_WITH_GRAD = 1
# 这个状态下，使用普通矩阵乘法，但不接受梯度更新。逐层DM化时，所有层都会先转到这个状态。
# 它可以提供DM训练层的反向传播阶梯，但不会更新参数。随后，第一层会转到DM backprop状态，接受梯度更新。
NIMBUS_STATE_MATMUL_NO_GRAD = 2
# DM的意思是differentiable MADDNESS。这个状态下，使用selection matrix、treeDesMatrix、thresholds等参数进行可微的MADDNESS操作,并接受梯度更新。
# 可微的过程使用了STE（Straight-Through Estimator）技术。主要更新两个参数：lut和thresholds。
# 
NIMBUS_STATE_DM_BACKPROP = 3
NIMBUS_STATE_MADDNESS_ONLY = 4
NIMBUS_STATE_DM_NO_GRAD = 5
#endregion

#region Consts:Usage

# several bitmask-like constants to specify what steps would be run. used for the training process type
# 1. normal training, which uses normal training process
STEP_LINEAR_ONLY = 1
# 2. replace each layer with differentiable MADDNESS layer, and retrain model iteratively
STEP_DIFFERENTIABLE_MADDNESS_LAYERS = 2
# 4-fine tuning the model
STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS = 4
# evaluate maddness-only model
STEP_EVALUATE_MADDNESS_ONLY = 8

