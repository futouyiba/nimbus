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
# 1. normal linear state, which uses normal mat mul and add operations
# 2. MADDNESS back-prop state, which uses matrices for tree-like operations, and can back-prop
# 3. MADDNESS only state, which uses direct tree-like operations, and cannot back-prop
NIMBUS_STATE_MATMUL = 1
NIMBUS_STATE_MADDNESS_BACKPROP = 2
NIMBUS_STATE_MADDNESS_ONLY = 3
NIMBUS_STATE_MATMUL_WITH_GRAD = 4
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

