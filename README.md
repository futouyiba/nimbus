# Cumulonimbus
An effective way to make neural network inferencing insanely fast while keep accuracy。

Cumulonimbus is the type of cloud that accumulates charge slowly and then discharges rapidly. The process involves the gradual build-up of electrical charges within the cloud, leading to a sudden and intense release of energy in the form of lightning.
Similarly, our project takes a little more time to train the model but once the model is trained, it can make inferences at a lightning-fast speed.

What's cooler is that we solve the problem of the accuracy degradation of approximate accelerated computing in multi-layer neural networks and provide a solution for it. Compared with previous works, our method has faster training speed and higher accuracy.
The most important thing is that we solve the problem that the training speed of differentiable MADDNESS drops sharply with the increase of the number of layers in multi-layer neural networks. This reduces the training time from O(exp(n)) to O(n).

一种有效的方法，使神经网络推理速度飞快，同时保持准确性。

积雨云是一种云的类型，它缓慢地积累电荷，然后迅速释放。这个过程涉及在云中逐渐积累电荷，导致以闪电形式的能量的突然和强烈的释放。
与此类似，我们的项目需要一点额外的时间来训练模型，但一旦模型训练好了，它就可以以闪电般的速度进行推理。

更酷的是，我们解决了近似加速计算在多层神经网络当中精度累积下降的问题，并为此提供了一个解决方案。相比前人的工作，我们的方法训练速度更快、精度更高。
其中最重要的是解决了可微的MADDNESS在多层神经网络中训练速度随着层数增加而急剧下降的问题。从而让训练时间从O(exp(n))降低到O(n)。

## Background
有两种场景对模型的推理速度有很高的要求：
1. 实时推理：比如游戏、网络包转发、自动驾驶、智能家居等场景，对模型的推理速度有很高的要求。
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