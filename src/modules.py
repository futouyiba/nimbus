import torch
import torch.nn as nn
from torch.nn import Parameter
import defines
import numpy as np


# create a diagonalized sparse matrix, which can be used to do selection of data
def create_selection_matrix(
        C: int = 1, K: int = 16, dtype=torch.float16
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
def create_4layer_tree_des_matrix(C: int = 1, dtype=torch.float16) -> torch.Tensor:
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

class NimbusLayer(nn.Module):
    count = 0

    def __init__(self):
        super(NimbusLayer, self).__init__()
        self.state = Parameter(torch.tensor(defines.NIMBUS_STATE_MATMUL, dtype=torch.int32), requires_grad=False)
        self.name = Parameter(torch.tensor(f'nimbus_layer_{NimbusLayer.count}', dtype=torch.string), requires_grad=False)
        self.register_load_state_dict_post_hook(self.state_dict_hook)
        NimbusLayer.count += 1

    def prepare_record_once(self) -> None:
        pass

    def state_dict_hook(self) -> None:
        pass

    def set_state(self, state: int) -> None:
        self.state = state


# an enhanced version of nn.Linear, which can handle MADDNESS features, and can switch between several states:
# 1. normal linear state, which uses normal mat mul and add operations
# 2. MADDNESS back-prop state, which uses matrices for tree-like operations, and can back-prop
# 3. MADDNESS only state, which uses direct tree-like operations, and cannot back-prop
# nn.Linear的增强版本，可以处理MADDNESS特性，并且可以在几种状态之间切换：
# 1. 正常线性状态，使用正常的矩阵乘法和加法操作
# 2. MADDNESS反向传播状态，使用树状操作的矩阵，可以反向传播
# 3. MADDNESS only状态，使用直接的树状操作，不能反向传播
class NimbusLinear(NimbusLayer, nn.Linear):
    def __init__(self, in_features, out_features, bias=True, state=defines.NIMBUS_STATE_MATMUL
                 , codeblockCount=-1, treeDepth=4):
        super(NimbusLinear, self).__init__()
        nn.Linear.__init__(self, in_features, out_features, bias)
        self.state = Parameter(torch.tensor(state, dtype=torch.int32), requires_grad=False)
        self.name = Parameter(torch.tensor(f'nimbus_linear_{NimbusLayer.count-1}', dtype=torch.string), requires_grad=False)
        self.treeDepth = Parameter(torch.tensor(treeDepth, dtype=torch.int32), requires_grad=False)
        self.bucketsPerBlock = 2 ** treeDepth
        # the LUT for MADDNESS
        self.lut = Parameter(torch.zeros(1, dtype=torch.bool), requires_grad=True)
        self.selectionMatrix = Parameter(torch.zeros(1), requires_grad=False)
        self.selectionMatrix = create_selection_matrix(C=codeblockCount, K=16)
        # how many codeblocks a row of input is divided into, noted as "C" in the maddness paper
        self.codeblockCount = Parameter(torch.tensor(codeblockCount, dtype=torch.int32), requires_grad=False)
        # the matrix describing the tree structure, use it to matmul could get value of bucket data should get into.
        # it's a 2D matrix, a diagonalized sparse matrix
        self.treeDesMat = Parameter(create_4layer_tree_des_matrix(), requires_grad=False)
        self.dims = Parameter(torch.zeros(1, dtype=torch.int32), requires_grad=False)
        self.thresholds = Parameter(torch.zeros(1, dtype=torch.float32), requires_grad=True)
        # a one time flag to record the input, used for saving the input for MADDNESS
        # the reason MADDNESS needs it is that buckets should be calculated based on the input, and LUT should be calculated based on the input
        self.want_to_record_once = False
        # self.split_factor = 1
        self.register_load_state_dict_post_hook(self.state_dict_hook)

    def forward(self, inputMatrix):
        # do a state switch
        if self.state == defines.NIMBUS_STATE_MATMUL:
            if self.want_to_record_once:
                # save the input for MADDNESS to a numpy file
                np.save(f'input_{self.name}.npy', inputMatrix.cpu().numpy())
                # one time only
                self.want_to_record_once = False
            return nn.functional.linear(inputMatrix, self.weight, self.bias).to(inputMatrix.device)
        elif self.state == defines.NIMBUS_STATE_MADDNESS_BACKPROP:
            # bunch of math ops here, Straight-Through Estimator (STE) is used to approximate the gradient
            # use selection matrix and thresholds to get the chosen elements
            # then do the tree-like operations using the treeDesMat
            # then do the approximate matmul with LUT
            chosen_elements = inputMatrix[:, self.dims]
            subtracted = self.selectionMatrix.mm(chosen_elements.T) - self.thresholds
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
        elif self.state == defines.NIMBUS_STATE_MADDNESS_ONLY:
            return self.forward_maddness_only(input)
        else:
            raise ValueError("Invalid state")

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
    def set_state(self, state: int) -> None:
        self.state = state
        if state == defines.NIMBUS_STATE_MADDNESS_BACKPROP:
            self.want_to_record_once = True


class NimbusConv1d(NimbusLayer, nn.Conv1d):
    pass