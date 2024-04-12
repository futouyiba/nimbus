import torch
from defines import *
import arg_settings


def train_procedure():
    model = arg_settings.TrainModelClass()
    model = model.to(device)
    model.train()

    #region training
    if STEP_LINEAR_ONLY & arg_settings.TrainProcessesInChain:
        # train the model
        pass
    if STEP_DIFFERENTIABLE_MADDNESS_LAYERS & arg_settings.TrainProcessesInChain:
        # replace each layer with differentiable MADDNESS layer, and retrain model iteratively
        pass
    if STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS & arg_settings.TrainProcessesInChain:
        # fine tune the model
        pass
    if STEP_EVALUATE_MADDNESS_ONLY & arg_settings.TrainProcessesInChain:
        # evaluate maddness-only model
        pass
    #endregion