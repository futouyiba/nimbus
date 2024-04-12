import os

CWD_PREFIX = "."

DATA_PATH = "data"
DATA_PATH_RAW = os.path.join(DATA_PATH, "raw")
LOGS_PATH = "logs"

# used for Halut layer state flags
# 1. normal linear state, which uses normal mat mul and add operations
# 2. MADDNESS back-prop state, which uses matrices for tree-like operations, and can back-prop
# 3. MADDNESS only state, which uses direct tree-like operations, and cannot back-prop
NIMBUS_STATE_NORMAL = 1
NIMBUS_STATE_MADDNESS_BACKPROP = 2
NIMBUS_STATE_MADDNESS_ONLY = 3

SAVES_PATH = "saves"
INPUT_CACHE_PATH = os.path.join(SAVES_PATH, "input_cache")
MODEL_CHECKPOINT_PATH = os.path.join(SAVES_PATH, "model_checkpoints")
RUNS_PATH = os.path.join(SAVES_PATH, "runs")

