import torch
import models
import argparse
import torch
import models
from os import path
import defines
from defines import STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS, STEP_DIFFERENTIABLE_MADDNESS_LAYERS, STEP_LINEAR_ONLY, STEP_EVALUATE_MADDNESS_ONLY
HiddenSize1 = 32
HiddenSize2 = 32
LearningRate = 0.01
BatchSize = 8192
EpochsEach = 80
WeightDecay = 0.0001
Momentum = 0.9
Device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
Optimizer = 'adam'
LRScheduler = 'cosine'
Criterion = 'cross_entropy'
Model = 'perceptron2'
Dataset = 'test_4'

TrainProcessesInChain = STEP_LINEAR_ONLY | STEP_DIFFERENTIABLE_MADDNESS_LAYERS | STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS | STEP_EVALUATE_MADDNESS_ONLY  # 1111
# TrainProcessesInChain = STEP_LINEAR_ONLY | STEP_DIFFERENTIABLE_MADDNESS_LAYERS | STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS   # 0111
InitialLearningRate = LearningRate

DataPath = path.join(defines.DATA_PATH_ROOT, Dataset)
TestData = torch.load(path.join(DataPath, 'test_data.pt'))
TestLabels = torch.load(path.join(DataPath, 'test_labels.pt'))
TestDataset = torch.utils.data.TensorDataset(TestData, TestLabels)
TestSampler = torch.utils.data.RandomSampler(TestDataset)
TestDataLoader = torch.utils.data.DataLoader(TestDataset, batch_size=BatchSize, sampler=TestSampler)
TrainData = torch.load(path.join(DataPath, 'train_data.pt'))
TrainLabels = torch.load(path.join(DataPath, 'train_labels.pt'))
TrainDataset = torch.utils.data.TensorDataset(TrainData, TrainLabels)
TrainSampler = torch.utils.data.RandomSampler(TrainDataset)
TrainDataLoader = torch.utils.data.DataLoader(TrainDataset, batch_size=BatchSize, sampler=TrainSampler)

target = TestLabels
print("target min", target.min(),"target max", target.max())

input_size = TestData.size(1)  # 获取testData的列数

# 获取testLabels当中的种类数
num_classes = len(torch.unique(target))

ModelInstance:models.NimbusModel   
if Model == 'perceptron2':
    ModelInstance = models.Perceptron2(input_size, num_classes, hidden_size1=HiddenSize1, hidden_size2=HiddenSize2)
else:
    ModelInstance = None

if Optimizer == 'adam':
    Optimizer = torch.optim.Adam(ModelInstance.parameters(), lr=InitialLearningRate, weight_decay=WeightDecay, betas=(Momentum, 0.999))
LRScheduler:torch.optim.lr_scheduler.LRScheduler
if LRScheduler == 'cosine':
    LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=EpochsEach, eta_min=0.0001)
else:
    LRScheduler = None

Criterion = torch.nn.CrossEntropyLoss()
