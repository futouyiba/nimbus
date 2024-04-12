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

#: 这个状态下，使用普通的矩阵乘法，慢但稳妥、精确、可反向传播。初始化模型时，所有层都是这个状态。
#: 这个状态会用来进行正常的训练，并得到一个基准模型。另外，存input也用这个状态进行。
NIMBUS_STATE_MATMUL_WITH_GRAD = 1

#: 这个状态下，使用普通矩阵乘法，但不接受梯度更新。逐层DM化时，所有层都会先转到这个状态。
#: 它可以提供DM训练层的反向传播阶梯，但不会更新参数。随后，第一层会转到DM backprop状态，接受梯度更新。
NIMBUS_STATE_MATMUL_NO_GRAD = 2

# DM的意思是differentiable MADDNESS。这个状态下，使用selection matrix、treeDesMatrix、thresholds等参数进行可微的MADDNESS操作,并接受梯度更新。
# 可微的过程使用了STE（Straight-Through Estimator）技术。主要更新两个参数：lut和thresholds。
# 在这个状态下，层的运算最慢。所以只在DM化和微调两个阶段使用这个状态。
NIMBUS_STATE_DM_BACKPROP = 3

# 使用类决策树+LUT的MADDNESS方法进行推导，不可微、不接受梯度。保留BN、激活层。
# 在一层DM化、重训完成后，将变为此状态，给下一层提供输出。后续可能会直接修改此处逻辑。
NIMBUS_STATE_MADDNESS_ONLY = 4

# 高度优化状态，在此状态下，使用MADDNESS推导、使用8位定点数、关闭BN、ReLU等层（用变换后LUT来提现这两层的作用），以达到最高的推导速度。
# 这个状态的实现依赖于nimbus层对于和它相关的BN层、激活层的引用、管理。
NIMBUS_STATE_HIGH_OPT = 10

# 这个状态只在开发、测试时使用，目的的是比对使用DM推导和MO（MADDNESS only）状态下对训练是否产生影响。
# 影响例如DM和MO的结果不一致、MO导致无法梯度降落等。实际训练时不应使用此状态。
NIMBUS_STATE_DM_NO_GRAD = -4

# 这个状态下，依然使用selection matrix、treeDesMatrix，因此可微、接受梯度，但使用8位定点数取代浮点、用
# 暂时不用。
NIMBUS_STATE_DM_OPT = -5
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

#endregion
