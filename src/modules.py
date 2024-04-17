import math
import os
import time
from typing import Literal
import torch
import torch.nn as nn
from torch.nn import Parameter,Linear
import defines
import numpy as np
import maddness
from maddness import Bucket, MultiSplit
import arg_settings


# create a diagonalized sparse matrix, which can be used to do selection of data
def create_selection_matrix(
        C: int = 1, K: int = 16, dtype=torch.float32
) -> torch.Tensor:
    depth = int(math.sqrt(K))
    selection_matrix = torch.zeros((C * 15, C * depth), dtype=dtype)
    based_selection_matrix = torch.zeros((K - 1, depth), dtype=dtype)
    for i in range(K - 1):
        if i == 0:
            based_selection_matrix[0, 0] = 1
        else:
            based_selection_matrix[i, int(np.log2(i + 1))] = 1
    for c in range(C):
        selection_matrix[
        c * 15: (c + 1) * 15, c * depth: (c + 1) * depth
        ] = based_selection_matrix
    return selection_matrix


# create a diagonalized sparse matrix, which can be used to do tree-like operations
# for now we only use 4 layer tree, with K = 16
def create_4layer_tree_des_matrix(C: int = 1, dtype=torch.float32) -> torch.Tensor:
    # example when using C = 1
    # fmt: off
    K = 16
    bit_matrix_numpy = np.array(
        [
            # 0
            [-1,
             -1, 0,
             -1, 0, 0, 0,
             -1, 0, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             -1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             1, 0, 0, 0,
             0, -1, 0, 0, 0, 0, 0, 0],
            [-1,
             -1, 0,
             1, 0, 0, 0,
             0, 1, 0, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, -1, 0, 0,
             0, 0, -1, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, -1, 0, 0,
             0, 0, 1, 0, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, 1, 0, 0,
             0, 0, 0, -1, 0, 0, 0, 0],
            [-1,
             1, 0,
             0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 0],
            # 8
            [1,
             0, -1,
             0, 0, -1, 0,
             0, 0, 0, 0, -1, 0, 0, 0],
            [1,
             0, -1,
             0, 0, -1, 0,
             0, 0, 0, 0, 1, 0, 0, 0],
            [1,
             0, -1,
             0, 0, 1, 0,
             0, 0, 0, 0, 0, -1, 0, 0],
            [1,
             0, -1,
             0, 0, 1, 0,
             0, 0, 0, 0, 0, 1, 0, 0],
            # 12
            [1,
             0, 1,
             0, 0, 0, -1,
             0, 0, 0, 0, 0, 0, -1, 0],
            [1,
             0, 1,
             0, 0, 0, -1,
             0, 0, 0, 0, 0, 0, 1, 0],
            [1,
             0, 1,
             0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, -1],
            [1,
             0, 1,
             0, 0, 0, 1,
             0, 0, 0, 0, 0, 0, 0, 1],
        ]
    )
    # fmt: on
    bit_matrix_base = torch.from_numpy(bit_matrix_numpy).to(dtype)
    bit_matrix = torch.ones((C * K, C * (K - 1)), dtype=dtype)
    for c in range(C):
        bit_matrix[
        c * K: (c + 1) * K,
        c * (K - 1): (c + 1) * (K - 1),
        ] = bit_matrix_base
    return bit_matrix

class NimbusLayer():
    count = 0

    def __init__(self):
        # super().__init__()
        # print("inner nimbus layer init")
        self.curComputeState = defines.NIMBUS_STATE_MATMUL_WITH_GRAD
        self.name:str = f'nimbus_layer_{NimbusLayer.count}'
        self.register_load_state_dict_post_hook(self.state_dict_hook)
        NimbusLayer.count += 1

    def prepare_record_once(self) -> None:
        pass

    def state_dict_hook(self) -> None:
        pass

    def set_state(self, state: int) -> None:
        self.state = state

    def learn_maddness_params(self) -> None:
        # TODO put logic here instead of in nimbus layer
        pass


# an enhanced version of nn.Linear, which can handle MADDNESS features, and can switch between several states:
# 1. normal linear state, which uses normal mat mul and add operations
# 2. MADDNESS back-prop state, which uses matrices for tree-like operations, and can back-prop
# 3. MADDNESS only state, which uses direct tree-like operations, and cannot back-prop
# nn.Linear的增强版本，可以处理MADDNESS特性，并且可以在几种状态之间切换：
# 1. 正常线性状态，使用正常的矩阵乘法和加法操作
# 2. MADDNESS反向传播状态，使用树状操作的矩阵，可以反向传播
# 3. MADDNESS only状态，使用直接的树状操作，不能反向传播
class NimbusLinear(Linear, NimbusLayer):
# class NimbusLinear(Linear, NimbusLayer):
    def __init__(self, in_features, out_features, bias=False, state=defines.NIMBUS_STATE_MATMUL_WITH_GRAD
                 , codeblockCount=-1, treeDepth=4):
        print("start nimbus linear init, in_features:", in_features, "out_features:", out_features, "bias:", bias, "state:", state, "codeblockCount:", codeblockCount, "treeDepth:", treeDepth)
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        # print("start nimbus layer init")
        NimbusLayer.__init__(self)
        # 下面的参数名应该可以解释他们自己是什么
        self.curComputeState = state
        self.name:str = f'nimbus_linear_{NimbusLayer.count-1}'
        self.treeDepth = Parameter(torch.tensor(treeDepth, dtype=torch.int32), requires_grad=False)
        bucketsPerBlock = 2 ** treeDepth
        self.bucketsPerBlock = Parameter(torch.tensor(bucketsPerBlock, dtype=torch.int32), requires_grad=False)
        # the LUT for MADDNESS
        self.lut = Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=True)

        if codeblockCount == -1:
            codeblockCount = in_features//2

        self.selectionMatrix:torch.Tensor = Parameter(create_selection_matrix(C=codeblockCount, K=16), requires_grad= True)
        # how many codeblocks a row of input is divided into, noted as "C" in the maddness paper
        self.codeblockCount = Parameter(torch.tensor(codeblockCount, dtype=torch.int32), requires_grad=False)
        # the matrix describing the tree structure, use it to matmul could get value of bucket data should get into.
        # it's a 2D matrix, a diagonalized sparse matrix
        self.treeDesMat = Parameter(create_4layer_tree_des_matrix(self.codeblockCount.item()), requires_grad=False)
        self.dims = Parameter(torch.zeros((self.selectionMatrix.shape[0],1), dtype=torch.int32), requires_grad=False)
        self.thresholds = Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=True)
        self.offset = Parameter(torch.Tensor([0.0]), requires_grad=False)
        self.scale = Parameter(torch.Tensor([1.0]), requires_grad=False)
        # a one time flag to record the input, used for saving the input for MADDNESS
        # the reason MADDNESS needs it is that buckets should be calculated based on the input, and LUT should be calculated based on the input
        self.want_to_record_once = False
        self.recorded_input:torch.Tensor = None
        # self.split_factor = 1
        # self.register_load_state_dict_post_hook(self.state_dict_hook)
        # TODO 这一层管理的其他层，例如BN层、激活层等，在做高度优化的时候，会被替换为LUT
        self.managed_layers = []
        self.all_splits = Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=False)

    def forward(self, inputMatrix):
        # do a state switch
        if self.curComputeState == defines.NIMBUS_STATE_MATMUL_WITH_GRAD:
            if self.want_to_record_once:
                self.recorded_input = inputMatrix
                # save the input for MADDNESS to a numpy file
                np.save(self.get_record_input_path(), inputMatrix.cpu().numpy())
                # one time only
                self.want_to_record_once = False
            return nn.functional.linear(inputMatrix, self.weight, self.bias).to(inputMatrix.device)
        elif self.curComputeState == defines.NIMBUS_STATE_DM_BACKPROP:
            # bunch of math ops here, Straight-Through Estimator (STE) is used to approximate the gradient
            # use selection matrix and thresholds to get the chosen elements
            # then do the tree-like operations using the treeDesMat
            # then do the approximate matmul with LUT
            chosen_elements:torch.Tensor = inputMatrix[:, self.dims]
            transposedChosen = chosen_elements.T
            subtracted = self.selectionMatrix.mm(transposedChosen) - self.thresholds
            tanh_h = torch.tanh(subtracted)
            sign_ste = torch.sign(subtracted) - tanh_h.detach() + tanh_h
            tree_result = self.treeDesMat.mm(sign_ste)
            tree_result = tree_result.T.reshape(-1, self.codeblockCount, self.bucketsPerBlock)
            encoding_soft = nn.Softmax(dim=2)(tree_result)
            index = torch.argmax(encoding_soft, dim=2, keepdim=True)
            encoding_hard = (torch.zeros_like(encoding_soft, memory_format=torch.legacy_contiguous_format)
                             .scatter_(2, index, 1.0))
            Encoded = encoding_hard - encoding_soft.detach() + encoding_soft
            out = torch.zeros([inputMatrix.shape[0], self.lut.shape[0]], dtype=torch.float32, device=inputMatrix.device)
            M = self.lut.size(0)
            out = torch.einsum("nij, kij -> nki", [Encoded, self.lut]).sum(dim=2)
            # put out tensor to the same device as inputMatrix
            out = out.to(inputMatrix.device)
        elif self.curComputeState == defines.NIMBUS_STATE_MADDNESS_ONLY:
            # TODO 优化、缓存、加速
            # 1. 缓存cpu上的numpy数组，用于MADDNESS的计算
            # 2. 在读取的时候作特殊处理
            # 3. （后续）将所有maddness的计算放到GPU上
            # encoded = maddness.halut_encode_opt(inputMatrix.cpu().numpy(), self.all_splits.cpu().numpy())
            # TODO 暂时使用matmul计算，先跑通锁层训练
            out = nn.functional.linear(inputMatrix, self.weight,self.bias).to(inputMatrix.device)
            # maddness.
            return self.forward_maddness_only(inputMatrix)
        else:
            raise ValueError("Invalid state")

        return out
    
    def forward_maddness_only(self, X):
        """
        使用纯maddness的方式计算输出，这种方式应当充分运用GPU并行加速，也应该是所有方法中计算最快的。（严谨的说应该慢于high opt模式，因为还有8位量化）
        X为输入矩阵，形状为（N，D），其中N是样本数，D是特征数。D=C*d，C是codeblockCount，d是单个codeblock的维数。
        每一行可以切分为C个codeblock，每个codeblock都可以视作一个决策树的输入向量，d维。
        self.dims形状为(C*depth),换句话说(1, C * 4)，
        self.lut形状为(C, K, M)，其中M是输出的维数，C是codeblockCount，K是bucketsPerBlock。
        """
        N, D = X.shape
        C = self.codeblockCount.item()
        d = D // C
        depth = self.treeDepth.item()
        K = self.bucketsPerBlock.item()
        
        # torch当中的weight是（M，D）的矩阵，其中M是输出的维数，D是输入的维数
        M = self.weight.shape[0]
        out = torch.zeros([N, M], dtype=torch.float32, device=X.device)

        # 计算所有决策树的结果
        # reshaping X to match (N, C, d)
        X_reshaped = X.view(N, C, d)
        # 获取每层的维度索引
        dim_indices = self.dims.view(1, C, depth).expand(N, C, depth)
        # 获取对应的阈值
        threshold_values = self.thresholds.view(1, C, K-1).expand(N, C, K-1)
        encoded = torch.ones(N, C, 1, dtype=torch.int64, device=X.device)
        lut_view = self.lut.view(C, K, M)

        for curDepth in range(depth):
            # 获取当前层的维度索引
            cur_dim_indices = dim_indices[:,:, curDepth]
            # 获取当前层的X数据
            cur_X = torch.gather(X_reshaped, 2, cur_dim_indices.unsqueeze(2).expand(N, C, 1))
            # 获取当前层的阈值
            cur_threshold_values = torch.gather(threshold_values, 2, encoded.expand(N, C, K-1))
            # 比较生成二进制决策结果
            decisions = cur_X < cur_threshold_values
            # 若小于，则将encoded对应的元素乘2，否则乘2再加1
            encoded = torch.where(decisions, encoded + encoded, encoded + encoded + torch.ones_like(encoded))
            
        # 使用encoded作为index，向self.lut中取值，然后求和，得到最终的输出
            # 生成用于gather的索引
        gather_indices = encoded.expand(-1, -1, M)
        # 从LUT中批量提取数据
        out = torch.gather(lut_view, 1, gather_indices).sum(dim=1)

        return out


    def state_dict_hook(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        if 'state' in state_dict:
            state_dict['state'] = state_dict['state'].item()
        if 'codeblockCount' in state_dict:
            state_dict['codeblockCount'] = state_dict['codeblockCount'].item()
        if 'treeDepth' in state_dict:
            state_dict['treeDepth'] = state_dict['treeDepth'].item()
        if 'dims' in state_dict:
            state_dict['dims'] = state_dict['dims'].cpu().numpy().tolist()
        if 'thresholds' in state_dict:
            state_dict['thresholds'] = state_dict['thresholds'].cpu().numpy().tolist()
        return state_dict
    
    def prepare_record_once(self):
        self.want_to_record_once = True

    # 这里类似一个状态机，根据状态的不同，进行不同的操作
    # 基本来说有4种状态
    def set_state(self, newState: int) -> None:
        print(f"{self.name} set state to {newState} from {self.curComputeState}")
        if newState == defines.NIMBUS_STATE_DM_BACKPROP:
            self.lut.requires_grad = True
            self.thresholds.requires_grad = True
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        elif newState == defines.NIMBUS_STATE_MADDNESS_ONLY:
            self.lut.requires_grad = False
            self.thresholds.requires_grad = False
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        elif newState == defines.NIMBUS_STATE_MATMUL_WITH_GRAD:
            self.lut.requires_grad = False
            self.thresholds.requires_grad = False
            self.weight.requires_grad = True
            if self.bias is not None:
                self.bias.requires_grad = True
        elif newState == defines.NIMBUS_STATE_MATMUL_NO_GRAD:
            self.lut.requires_grad = False
            self.thresholds.requires_grad = False
            self.weight.requires_grad = False
            if self.bias is not None:
                self.bias.requires_grad = False
        elif newState == defines.NIMBUS_STATE_DM_OPT:
            pass
        
        if self.curComputeState == defines.NIMBUS_STATE_MATMUL_WITH_GRAD and newState == defines.NIMBUS_STATE_DM_BACKPROP:
            self.learn_maddness_params(self.recorded_input.cpu().numpy())

        if self.curComputeState == defines.NIMBUS_STATE_DM_BACKPROP and newState != defines.NIMBUS_STATE_DM_BACKPROP:
            # do some cleanup
            # state transition specific logic here
            pass

        self.curComputeState = newState



    def learn_maddness_params(self, X):
        """
        Learn the madness parameters.

        Parameters:
        - X: The input data matrix of shape (N, D), where N is the number of samples and D is the number of features.

        Returns:
        - None

        This method calculates the madness parameters based on the input data matrix X.

        学习MADDNESS参数。

        参数：
        - X：形状为（N，D）的输入数据矩阵，其中N是样本数，D是特征数。

        返回：
        - 无

        此方法根据输入数据矩阵X以及self的成员变量计算MADDNESS参数。
        """
        all_splits_np, all_prototypes, _, thresholds, dims = maddness.learn_proto_and_hash_function(X, self.codeblockCount.item(), self.bucketsPerBlock.item())
        self.dims.data = torch.from_numpy(dims).to(arg_settings.Device)
        self.thresholds.data = torch.from_numpy(thresholds).unsqueeze(1).to(arg_settings.Device)
        # lut_numpy = maddness.maddness_lut(self.weight.cpu(), all_prototypes= all_prototypes)
        B = self.weight.cpu().numpy()
        lut_numpy = np.zeros((B.shape[0], self.codeblockCount.item(), self.bucketsPerBlock.item()))
        for i, q in enumerate(B):
            lut_numpy[i] = maddness.maddness_lut(q, all_prototypes)
        self.lut.data = torch.from_numpy(lut_numpy).float().to(arg_settings.Device)
        self.all_splits.data = torch.from_numpy(all_splits_np).to(arg_settings.Device)
        
    def load_recorded_input(self):
        self.recorded_input = np.load(self.get_record_input_path())

    def record_input(self, inputMatrix):
        """
            缓存输入矩阵，目的是为了学习MADDNESS参数，一般发生在预训练完毕、DM化开始之前。
            缓存在内存里做一份，因为成本很低，且有可能后续直接衔接DM化，不需要再次加载。
            在磁盘上做两份，一份是加了时间戳/序号的，用来事后检视,一份是latest,用来读取(因为读取时往往不知道上一次的时间戳/序号)
        """
        self.recorded_input = inputMatrix
        np.save(self.get_record_input_path(), inputMatrix.cpu().numpy())

    
    def get_record_input_path(self):
        """
            拼接输入缓存文件的路径
            使用的场景比较特殊，就在上一次做了预训练，但是没有继续往下做DM化，又或者DM化只持续了几层，然后就停止了。
            于是我们重新开启了训练，这时候我们就需要把上一次的输入缓存文件加载进来，然后继续往下做DM化。
            如果是一次性从预训练跑到DM化，那recorded_input就已经在内存对象里了，不需要加载。
        """
        return os.path.join(defines.INPUT_CACHE_PATH_ROOT, f'input_{self.name}_latest.npy')

class NimbusConv1d(NimbusLayer, nn.Conv1d):
    pass
