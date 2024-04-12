import models
from defines import *

#region Args:Usage

TrainModelClass = models.Perceptron2
TrainProcessesInChain = STEP_LINEAR_ONLY | STEP_DIFFERENTIABLE_MADDNESS_LAYERS | STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS | STEP_EVALUATE_MADDNESS_ONLY  # 1111
#endregion
