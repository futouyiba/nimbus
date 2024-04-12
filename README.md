# Cumulonimbus
An effective way to make neural network inferencing insanely fast while keep accuracy。

Cumulonimbus is the type of cloud that accumulates charge slowly and then discharges rapidly. The process involves the gradual build-up of electrical charges within the cloud, leading to a sudden and intense release of energy in the form of lightning.
Similarly, our project takes a little more time to train the model but once the model is trained, it can make inferences at a lightning-fast speed.

What's cooler is that compared with previous works, our method has faster training speed and higher accuracy.
We've done several improvement, among which we're most proud of that we solve the problem that the training speed of differentiable MADDNESS drops sharply with the increase of the number of layers in multi-layer neural networks. This reduces the training time from O(exp(n)) to O(n).

一种有效的方法，使神经网络推理速度飞快，同时保持准确性,而训练过程也时间成本可控。

积雨云是一种云的类型，它缓慢地积累电荷，然后迅速释放。这个过程涉及在云中逐渐积累电荷，导致以闪电形式的能量的突然和强烈的释放。
与此类似，我们的项目需要一点额外的时间来训练模型，但一旦模型训练好了，它就可以以闪电般的速度进行推理。

更酷的是，相比前人的工作，我们的方法训练速度更快、精度更高。
我们达成了多项优化，其中最重要的是解决了可微的MADDNESS在多层神经网络中训练速度随着层数增加而急剧下降的问题。从而让训练时间从O(exp(n))降低到O(n)。

## Background
有几种场景对模型的推理速度有很高的要求：
1. 实时推理：比如游戏、网络包转发、自动驾驶、智能家居等场景，对模型的推理速度有很高的要求。
1. 嵌入场景：比如在网络硬件、IOT设备当中，计算资源
2. 大模型：模型大了之后，推理速度会变得很慢，比如BERT、GPT等模型。用户希望能够在保证模型精度的情况下，提高模型的推理速度。 

近年来围绕D Blalock的论文《Multiplying Matrices without Multiplying》的研究，使用LUT（Look-Up Table）来加速矩阵乘法的计算，取得了很好的效果。我们希望能够将这种方法应用到神经网络的推理过程中，提高神经网络的推理速度。


There are two scenarios that have high requirements for the inference speed of the model:
1. Real-time inference: For example, games, network packet forwarding, autonomous driving, smart home and other scenarios have high requirements for the inference speed of the model.
2. Large model: After the model becomes large, the inference speed will become very slow, such as BERT, GPT and other models. Users hope to improve the inference speed of the model while ensuring the accuracy of the model.

## Inspirations
1. D Blalock, Multiplying Matrices without Multiplying, 2018
2. Stella Nera
3. LUT-NN

## Paper
To be published

## Usage
TBW

## Design 设计
### Ideas 思考
是否应该只用矩阵乘法的神经网络进行训练，然后直接将其MADDNESS化？
- 不应该。经过实验，MADDNESS带来的准确率损失会逐层累乘，导致神经网络的表现无法接受
是否应该直接使用Differentiable MADDNESS构建神经网络，直接开始训练？
- 不应该。经过探索，我们发现使用differentiable MADDNESS（下文简称为MD）进行训练有非常大的性能消耗，在层数我们发现分桶（MADDNESS概念）、LUT初始值都需要一个表现还不错的神经网络提供支持。
如何让训练过程不严重失速？
### Procedures 过程
训练过程分为几个阶段：
- 初训。首先使用普通的矩阵乘法、神经网络、反向传播，来获得一个训练基准。这个阶段主要达成几个目的：
-- 给DM的LUT优化提供input
-- 给MADDNESS的分桶提供参数矩阵（可看成MADDNESS论文中的B矩阵）。在神经网络的背景下，参数矩阵通常指的是weight矩阵
- 逐层DM化，并锁住其他层进行单层训练。

  这个过程中，Nimbus Layer在几种状态中来回切换，以达到训练目的。这是一个对于Nimbus Layer的enumerate过程，其中的每一步对一个Nimbus Layer进行转训，基本操作为：
  - 将上一个转训层切换状态，使用高速的MADDNESS算法进行推理，输出数据给当前Nimbus Layer作为输入使用
  - 
-- 
