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

from os import path
import time
import torch
from defines import *
import arg_settings
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from models import NimbusModel
from modules import NimbusLayer, NimbusLinear
import subprocess
import atexit
import subprocess
from os import path
import os


class TrainProcedure:
    """把训练过程封装成一个类，方便管理。
        这个类使用单例模式，让其他类都能访问到这个类的实例。

    """

    # 单例模式
    _instance = None
    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(TrainProcedure, cls).__new__(cls)
        return cls._instance
    
    @classmethod
    def get_instance(cls):
        if not cls._instance:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        # TODO put these into function body instead of global variables
        self.LastDMizedLayer: NimbusLayer = None
        self.CurDMizedLayer: NimbusLayer = None

        self.GlobalBestAccuracy = 0.0
        self.StepBestAccuracy = 0.0
        self.GlobalBestCheckpointPath = None
        self.StepBestCheckpointPath = None
        # 当前进行到的步骤。1为普通训练，2为DM化，3为微调，4为评估MADDNESS only。其中特殊的是DM化，它是一个逐层的过程，每一层都会进行一次DM化。这时会加上0.01
        # 举例：2.01为第一层DM化，2.02为第二层DM化，以此类推。
        self.curStep = 0.0
        self.curEpoch = 0

        self.modelDataCombined = f"{arg_settings.ModelInstance.__class__.__name__}-{arg_settings.DatasetName}"
        self.procedure_start_time = time.strftime("%m%d%H%M", time.localtime(time.time()))
        self.procedure_name = f"{self.modelDataCombined}-{arg_settings.runCodeName}-{self.procedure_start_time}"
        self.runPath = f'{RUNS_PATH_ROOT}/{self.procedure_name}'
        self.writer = SummaryWriter(self.runPath)
        self.checkpointFolder = path.join(MODEL_CHECKPOINT_PATH_ROOT, self.modelDataCombined)
        if not path.exists(self.checkpointFolder):
            os.makedirs(self.checkpointFolder)
        self.checkpointLatestPath = path.join(self.checkpointFolder, f'{arg_settings.runCodeName}-latest.pth')

        # 启动TensorBoard
        # self.tensorboard_process = subprocess.Popen(['python', '-m', 'tensorboard.main', '--logdir', self.runPath])
        self.nimbus_layers = arg_settings.ModelInstance.get_nimbus_layers()


    def SaveInputCache(self, model: nn.Module):
        """
        Saves the input cache for the given model.

        Args:
            model (nn.Module): The model to save the input cache for.

        Returns:
            None
        """
        model.eval()
        for l in self.nimbus_layers:
            l.prepare_record_once()
        with torch.inference_mode():
            model(arg_settings.TrainData.to(arg_settings.Device))


    def train_epochs(self, model: nn.Module, epochs:int):
        start_epoch = self.curEpoch + 1
        print(f"Training model {model.__class__.__name__} for {epochs} epochs, starting from epoch {start_epoch}.")
        self.StepBestAccuracy = 0.0
        self.StepBestCheckpointPath = None
        self.GlobalBestCheckpointPath = None
        trainStepStartTime = time.time()
        for epoch in range(start_epoch, start_epoch+epochs):
            epoch_start_time = time.time()
            self.curEpoch = epoch
            print(f"Epoch {epoch} started.")
            epoch_training_loss = 0.0
            correct = 0
            total = 0
            model.train()
            arg_settings.Optimizer.zero_grad()
            # summary writer写入这个epoch当前的（经过lr decay的）learning rate
            self.writer.add_scalar('crucial/learning_rate', arg_settings.Optimizer.param_groups[0]['lr'], epoch)
            # batch training
            for i, (data, target) in enumerate(arg_settings.TrainDataLoader):
                data, target = data.to(arg_settings.Device), target.to(arg_settings.Device)
                output = model(data)
                loss = arg_settings.Criterion(output, target)
                epoch_training_loss += loss.item()
                loss.backward()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                arg_settings.Optimizer.step()
                arg_settings.Optimizer.zero_grad()
                acc = (output.argmax(dim=1) == target).float().mean()
                # 记录每个batch的loss
                self.writer.add_scalar('batches/train_acc', acc, epoch * len(arg_settings.TrainDataLoader) + i)
            acc = 100 * correct / total
            self.writer.add_scalar('epoch/training_loss', epoch_training_loss, epoch)
            self.writer.add_scalar('crucial/training_acc', acc, epoch)
            epoch_end_time = time.time()
            elapsed_time = epoch_end_time - epoch_start_time
            self.writer.add_scalar('crucial/epoch_time', elapsed_time, epoch)
            # evaluate on test set
            test_acc = self.evaluate_model(model, epoch=epoch)
            arg_settings.LRScheduler.step()
            print(f"Epoch {epoch} finished. Test accuracy: {test_acc}")
        print(f"Training {self.curStep:.2f} finished. Time elapsed: {time.time()-trainStepStartTime} seconds.")
        # TODO 把这些路径整理一下。现在先专注于功能，后续再整理路径。
        # 重置scheduler & optimizer
        arg_settings.Optimizer = torch.optim.Adam(model.parameters(), lr=arg_settings.InitialLearningRate, weight_decay=arg_settings.WeightDecay, betas=(arg_settings.Momentum, 0.999))
        arg_settings.LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(arg_settings.Optimizer, T_max=arg_settings.EpochsEach, eta_min=0.0001)

    def evaluate_model(self, model: nn.Module, epoch:int,)->float:
        model.eval()
        with torch.inference_mode():
            correct = 0
            total = 0
            epoch_test_loss = 0.0
            for data, target in arg_settings.TestDataLoader:
                data, target = data.to(arg_settings.Device), target.to(arg_settings.Device)
                output = model(data)
                loss = arg_settings.Criterion(output, target)
                epoch_test_loss += loss.item()
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
            acc = 100 * correct / total
            self.writer.add_scalar('crucial/test_acc', acc, epoch)
            self.writer.add_scalar('epoch/test_loss', epoch_test_loss, epoch)

            if(acc>self.StepBestAccuracy):
                self.StepBestAccuracy = acc
                if self.StepBestCheckpointPath is not None:
                    os.remove(self.StepBestCheckpointPath)
                self.StepBestCheckpointPath = f"{MODEL_CHECKPOINT_PATH_ROOT}/{arg_settings.ModelInstance.__class__.__name__}/{arg_settings.runCodeName}_step{self.curStep:.1f}_best_{epoch}_{acc:.2f}.pth"
                torch.save(model.state_dict(), self.StepBestCheckpointPath)
            if(acc>self.GlobalBestAccuracy):
                self.GlobalBestAccuracy = acc
                if self.GlobalBestCheckpointPath is not None:
                    os.remove(self.GlobalBestCheckpointPath)
                self.GlobalBestCheckpointPath = f"{MODEL_CHECKPOINT_PATH_ROOT}/{arg_settings.ModelInstance.__class__.__name__}/{arg_settings.runCodeName}_global_best_{epoch}_{acc:.2f}.pth"
                torch.save(model.state_dict(), self.GlobalBestCheckpointPath)
            return acc

    # 把模型中某一层替换为可微的MADDNESS层，锁定其他层不进行更新。这个过程supposedly被逐个使用，比如一个3层网络，在第一层DM化（differentiable MADDNESS化）之后，
    # 第一层以MADDNESS only的方式进行推导，且不接受梯度更新，第二层以DM的方式进行推导，且接受梯度更新，第三层以普通mm的方式进行推导，不接受梯度更新。
    # 这样保持了一定的速度，又能逐层的进行DM化，完成量化后重训练。
    # Substitutes a layer in the model with a differentiable MADDNESS layer, and locks other layers for gradient update.
    # This process is supposed to be used iteratively, for example, a 3-layer network, after the first layer is DM-ized (differentiable MADDNESS-ized),
    # the first layer is derived in MADDNESS only mode, and does not accept gradient updates, the second layer is derived in DM mode, and accepts gradient updates,
    # the third layer is derived in normal mm mode, and does not accept gradient updates.
    # This way, a certain speed is maintained, and the DM-ization is done layer by layer, and retraining is done after quantization.
    def DM_next_layer(self, model: NimbusModel)->bool:
        nimbusLayers = model.get_nimbus_layers()
        
        if self.LastDMizedLayer is None:
            self.CurDMizedLayer = nimbusLayers[0]
        else:
            self.LastDMizedLayer = self.CurDMizedLayer
            if nimbusLayers.index(self.LastDMizedLayer) == len(nimbusLayers) - 1:
                return False
            self.CurDMizedLayer = nimbusLayers[nimbusLayers.index(self.LastDMizedLayer) + 1]

        self.LastDMizedLayer.set_state(NIMBUS_STATE_MADDNESS_ONLY)
        self.CurDMizedLayer.set_state(NIMBUS_STATE_DM_BACKPROP)


    def lock_all_nimbus_layers(self, model: NimbusModel):
        nimbusLayers = model.get_nimbus_layers()
        for i in range(len(nimbusLayers)):
            layer = nimbusLayers[i]
            # prevent gradient update
            for param in layer.parameters():
                param.requires_grad = False

    def nimbus_layer_require_grad(self, layer: NimbusLinear):
        layer.lut.requires_grad = True
        layer.thresholds.requires_grad = True
        layer.bias.requires_grad = True
        layer.weight.requires_grad = True

    def start(self):
        model = arg_settings.ModelInstance  # Assign the value of arg_settings.TrainModel to the variable model
        model = model.to(arg_settings.Device)
        model.train()

        nimbusLayers = model.get_nimbus_layers()

        #region training
        if STEP_LINEAR_ONLY & arg_settings.TrainProcessesInChain:
            # train the model
            self.curStep = 1
            self.train_epochs(model, arg_settings.EpochsEach)
            self.SaveInputCache(model)
        if STEP_DIFFERENTIABLE_MADDNESS_LAYERS & arg_settings.TrainProcessesInChain:
            self.curStep = 2
            # replace each layer with differentiable MADDNESS layer, and retrain model iteratively
            for l in nimbusLayers:
                l.set_state(NIMBUS_STATE_MATMUL_WITH_GRAD)

            for l in nimbusLayers:
                self.CurDMizedLayer = l
                l.set_state(NIMBUS_STATE_DM_BACKPROP)
                if self.LastDMizedLayer is not None:
                    self.LastDMizedLayer.set_state(NIMBUS_STATE_MADDNESS_ONLY)
                # self.DM_next_layer(model)
                self.train_epochs(model, arg_settings.EpochsEach)
                self.LastDMizedLayer = l
                self.curStep += 0.01
        if STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS & arg_settings.TrainProcessesInChain:
            self.curStep = 3
            # fine tune the model
            arg_settings.Optimizer = torch.optim.Adam(model.parameters(), lr=arg_settings.InitialLearningRate, weight_decay=arg_settings.WeightDecay, betas=(arg_settings.Momentum, 0.999))
            arg_settings.LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(arg_settings.Optimizer, T_max=arg_settings.EpochsEach, eta_min=0.0001)
            for l in nimbusLayers:
                l.set_state(NIMBUS_STATE_DM_BACKPROP)
            self.train_epochs(model, 90)
        if STEP_EVALUATE_MADDNESS & arg_settings.TrainProcessesInChain:
            self.curStep = 4
            # evaluate maddness-only model
            for l in nimbusLayers:
                l.set_state(NIMBUS_STATE_RSLT_CHK)
            self.evaluate_model(model, 1)
        #endregion
        # if exist, remove the latest checkpoint file and save the latest model
        if path.exists(self.checkpointLatestPath):
            os.remove(self.checkpointLatestPath)
        torch.save(arg_settings.ModelInstance.state_dict(), self.checkpointLatestPath)


    def cleanup(self):
        self.writer.close()
        torch.save(arg_settings.ModelInstance.state_dict(), self.checkpointLatestPath)
        # self.tensorboard_process.terminate()


if __name__ == "__main__":
    trainInst = TrainProcedure()
    atexit.register(trainInst.cleanup)
    trainInst.start()

    print("=====================================================================================")
    print("Training finished.")
    print("=====================================================================================")