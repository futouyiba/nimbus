import torch
import models
import argparse
import torch
import models
from os import path
import defines
from defines import STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS, STEP_DIFFERENTIABLE_MADDNESS_LAYERS, STEP_LINEAR_ONLY, STEP_EVALUATE_MADDNESS_ONLY

parser = argparse.ArgumentParser()

# Add arguments for training settings
parser.add_argument('--hidden_size1', type=int, default=32, help='Size of the first hidden layer')
parser.add_argument('--hidden_size2', type=int, default=32, help='Size of the second hidden layer')
parser.add_argument('--learning_rate', type=float, default=0.01, help='Initial learning rate')
parser.add_argument('--batch_size', type=int, default=8192, help='Batch size')
parser.add_argument('--epochs', type=int, default=80, help='Number of epochs')
parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device to use')
parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer')
parser.add_argument('--lr_scheduler', type=str, default='cosine', help='Learning rate scheduler')
parser.add_argument('--criterion', type=str, default='cross_entropy', help='Loss criterion')
parser.add_argument('--model', type=str, default='perceptron2', help='Model to use')
parser.add_argument('--dataset', type=str, default='test_4', help='Dataset to use')

args = parser.parse_args()

TrainModel:models.NimbusModel   
if args.model == 'perceptron2':
    TrainModel = models.Perceptron2(14, 3, hidden_size1=args.hidden_size1, hidden_size2=args.hidden_size2)
else:
    TrainModel = None
TrainProcessesInChain = STEP_LINEAR_ONLY | STEP_DIFFERENTIABLE_MADDNESS_LAYERS | STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS   # 0111
InitialLearningRate = args.learning_rate
BatchSize = args.batch_size
EpochsEach = args.epochs
WeightDecay = args.weight_decay
Momentum = args.momentum
Device = torch.device(args.device)
if args.optimizer == 'adam':
    Optimizer = torch.optim.Adam(TrainModel.parameters(), lr=InitialLearningRate, weight_decay=WeightDecay, betas=(Momentum, 0.999))
if args.lr_scheduler == 'cosine':
    LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=EpochsEach, eta_min=0.0001)
else:
    LRScheduler = None

Criterion = torch.nn.CrossEntropyLoss()

DataPath = path.join(defines.DATA_PATH_ROOT, args.dataset)
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
