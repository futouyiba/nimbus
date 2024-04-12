import torch
import models
from defines import *
import argparse
import torch
import models
from defines import *

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

args = parser.parse_args()

TrainModel = models.Perceptron2(14, 3, hidden_size1=args.hidden_size1, hidden_size2=args.hidden_size2)
TrainProcessesInChain = STEP_LINEAR_ONLY | STEP_DIFFERENTIABLE_MADDNESS_LAYERS | STEP_FINE_TUNE_DIFFERENTIABLE_MADDNESS | STEP_EVALUATE_MADDNESS_ONLY  # 1111
InitialLearningRate = args.learning_rate
BatchSize = args.batch_size
EpochsEach = args.epochs
WeightDecay = args.weight_decay
Momentum = args.momentum
Device = torch.device(args.device)
Optimizer = torch.optim.Adam(TrainModel.parameters(), lr=InitialLearningRate, weight_decay=WeightDecay, betas=(Momentum, 0.999))
LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(Optimizer, T_max=EpochsEach, eta_min=0.0001)
Criterion = torch.nn.CrossEntropyLoss()
