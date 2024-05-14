# Cumulonimbus
An effective way to make neural network inferencing insanely fast while keep accuracy。

Cumulonimbus is the type of cloud that accumulates charge slowly and then discharges rapidly. The process involves the gradual build-up of electrical charges within the cloud, leading to a sudden and intense release of energy in the form of lightning.
Similarly, our project takes a little more time to train the model but once the model is trained, it can make inferences at a lightning-fast speed.

What's cooler is that compared with previous works, our method has faster training speed and higher accuracy.
We've done several improvement, among which we're most proud of that we solve the problem that the training speed of differentiable MADDNESS drops sharply with the increase of the number of layers in multi-layer neural networks. This reduces the training time from O(exp(n)) to O(n).

一种有效的方法，使神经网络推理速度飞快，同时保持准确性,而训练过程也时间成本可控。在对应的硬件中可以规避乘法器的使用。

积雨云是一种云的类型，它缓慢地积累电荷，然后迅速释放。这个过程涉及在云中逐渐积累电荷，导致以闪电形式的能量的突然和强烈的释放。
与此类似，我们的项目需要一点额外的时间来训练模型，但一旦模型训练好了，它就可以以闪电般的速度进行推理。

更酷的是，相比前人的工作，我们的方法训练速度更快、精度更高。
我们达成了多项优化，其中最重要的是解决了可微的MADDNESS在多层神经网络中训练速度随着层数增加而急剧下降的问题。从而让训练时间从O(exp(n))降低到O(n)（注：n<100（TODO）时）。

## Background
机器学习、深度学习正在各种场景下显著的帮助人们。
而矩阵乘法是位于它们中心的基石。由于它与计算机软硬件的亲和性，以及人类数学体系中对矩阵的大量研究，它帮助机器学习、深度学习取得了许多重要进展。
然而，对于矩阵乘法的速度，人们逐渐不满足。

尤其是有几种场景对模型的推理速度有很高的要求：
1. 实时推理：比如游戏、网络包转发、自动驾驶等场景，对模型的推理速度有很高的要求。
1. 嵌入场景：比如在网络硬件、IOT设备当中，计算资源有限，且计算资源在可见的将来不会有明显增加。
2. 大模型：模型大了之后，推理速度会变得很慢，比如BERT、GPT等模型。用户希望能够在保证模型精度的情况下，提高模型的推理速度。 


There are two scenarios that have high requirements for the inference speed of the model:
1. Real-time inference: For example, games, network packet forwarding, autonomous driving, smart home and other scenarios have high requirements for the inference speed of the model.
2. Large model: After the model becomes large, the inference speed will become very slow, such as BERT, GPT and other models. Users hope to improve the inference speed of the model while ensuring the accuracy of the model.

## Former Work
### Strassen方法 TODO 检查是否是这个名字
核心的思路是“分而治之”和“剪除冗余”。即将大的矩阵运算拆分为若干个子矩阵，进行分别运算后的聚合。以及 TODO 剪除冗余。

### Vector Quantization
TODO
### Bolt
TODO<sup>[1]</sup>

### Model Quantization

### Binary Neural Network（BNN）

### Product Quantization
Product Quantization（PQ）的主要思路是“就近分配”：

- 将A矩阵划分为若干个子空间
- 提前在每个子空间中找到若干个质心
- 将质心和B矩阵对应片段（slices）的乘积结果保存为LUT
- 计算时，将各个codeblock“就近分配”到对应的质心，根据分配到的质心index查找LUT
- 进行对应的sum，得到计算结果
TODO：PQ的图


### MADDNESS
近年来围绕Davis Blalock的论文《Multiplying Matrices without Multiplying》的研究，使用LUT（Look-Up Table）来加速矩阵乘法的计算，取得了很好的效果。我们希望能够将这种方法应用到神经网络的推理过程中，提高神经网络的推理速度。

MADDNESS的主要思路是用树判断代替“就近”的计算。这个代替的思路参考了“Local-Sensitive Hashing”。

TODO 放图

MADDNESS的算法本身是通用的，但其中包含了几个关键性的优化，这些优化主要针对CPU场景设计。

### MADDNESS番外
最终，我们发现MADDNESS推动了线性计算和决策树的互相借助、互相融合：

在以往的实践中，决策树一直以其运算高效、训练速度快而深受青睐，甚至在其上衍生出了随机森林等算法。

但是决策树较为依赖特征工程。而在特征工程方向不明确时，MLP可以帮助研究者找到隐含的特征关系。

而在其中的线性计算变成了算法的主要部分时，计算速度就开始成为问题。

此时用决策树+LUT去逼近线性计算，就得到了更快的运算速度，可以说是线性计算的决策树版平替。

所以后续有可能催化出这样的发展：在高度需要灵活性的地方，把决策树的一些部分线性计算化；在高度需要计算速度的地方，把神经网络的一些部分决策树化。

### LUT-NN
核心思路是通过STE方法来让PQ变得可微，并通过retraining步骤，让PQ的累加误差得到大幅度降低。
TODO 图

### Stella Nera
核心思路是通过STE方法，让MADDNESS变得可微、可训练。并将它硬件化，得到了很高的计算效率。

## Our Approach
核心思路是：
- 让算法亲和GPU，以贴近实际应用场景
- 让神经网络的MADDNESS化流程容易、清晰、高效
- 改进分桶、LUT算法，提升效果

通过以上方法，nimbus可以帮助更常见的神经网络达到高效率，同时不丢失太多精确度。
### 围绕GPU设计算法
#### 更好的树判断
决策树的一层可用不同元素作为条件进行判断
#### 训练、推理全面GPU化
基于上述改进，提供了MADDNESS以及differentiable MADDNESS的GPU版实现
### 联合分桶
通过将bucketing过程和B矩阵对应的片段结合起来，我们将NMSE降低，得到了更好的重训练初始值。
### 策略模式
NimbusLayer层内部包含

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
### Computing 运算
Cumulonimbus程序的核心是NimbusLayer，它可以在几种运算模式中切换。具体可见《modules.py》 TODO 加入文件链接

#### MADDNESS模式
MADDNESS模式的计算方式与《Multiplying Matrices Without Multiplying》中一样
#### DM模式（Differential MADDNESS）（可微的MADDNESS）
 **使用爱因斯坦矩阵和**而非对角化
### Procedures 过程
训练过程分为几个阶段：
- 初训。首先使用普通的矩阵乘法、神经网络、反向传播，来获得一个训练基准。这个阶段主要达成几个目的：
-- 给DM的LUT优化提供input
-- 给MADDNESS的分桶提供参数矩阵（可看成MADDNESS论文中的B矩阵）。在神经网络的背景下，参数矩阵通常指的是weight矩阵
- DM化（Differentiable MADDNESSize）

  逐层DM化，并锁住其他层进行单层训练。这个过程中，Nimbus Layer在几种状态中来回切换，以达到训练目的。这是一个对于Nimbus Layer的enumerate过程，对其中的第l层，基本操作为：
  - 将l-1层切换状态（l==1的处理省略），使用高速的MADDNESS算法进行推理，输出数据给第l层作为输入使用
  - 将l层切换状态，使用selection matrix、tree des matrix来进行可微的MADDNESS计算并进行梯度降落，以使得
  - 后面的L-l层都锁住，使用普通MatMul方法计算、传递梯度，但不做梯度降落
- 量化（前期跳过）。使用8位定点数代替浮点数。
- 微调。这个阶段里，每个Nimbus Layer都植入DM状态，进行较为缓慢的DM梯度降落。这一步的成本由于前几步而降低，尤其是在当前主流的深度学习现状下，典型的层数>8，<100(TODO)，这时受益显著。
- 终优化。使用变换后的LUT取代BN层、激活层等，达到进一步的速度提升。