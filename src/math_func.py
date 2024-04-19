import math
import torch

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
    bit_matrix_base = torch.tensor(
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
        ],
        dtype=dtype
    )

    bit_matrix = torch.ones((C * K, C * (K - 1)), dtype=dtype)
    for c in range(C):
        bit_matrix[
        c * K: (c + 1) * K,
        c * (K - 1): (c + 1) * (K - 1),
        ] = bit_matrix_base
    return bit_matrix

def _cumsse_cols_torch(X):
    N, D = X.shape
    # 计算累积和与累积平方和
    cumX_column = torch.cumsum(X, dim=0)
    cumX2_column = torch.cumsum(X * X, dim=0)

    # 计算每一行的均值
    counts = torch.arange(1, N+1, dtype=torch.float32).unsqueeze(1)  # 转换为列向量
    meanX = cumX_column / counts

    # 计算累积平方误差
    cumsses = cumX2_column - (cumX_column * meanX)

    # 第一行错误为0
    cumsses[0, :] = 0

    return cumsses

def optimal_split_val_torch(X, dim, X_orig=None):
    if X_orig is None:
        X_orig = X
    assert X_orig.shape == X.shape

    if X.shape[0] == 0:
        return dim, 0, float('inf')

    # 排序
    sort_idxs = torch.argsort(X_orig[:, dim])
    X_sort = X[sort_idxs]

    # 计算累积平方误差
    sses_head = _cumsse_cols_torch(X_sort)
    sses_tail = _cumsse_cols_torch(X_sort.flip(dims=[0])).flip(dims=[0])
    sses = sses_head
    sses[:-1] += sses_tail[1:]
    sses_total = sses.sum(dim=1)

    # 找到最小 SSE 的索引
    best_idx = torch.argmin(sses_total)
    next_idx = min(X.shape[0] - 1, best_idx + 1)
    col = X[:, dim]
    best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2

    return dim, best_val.item(), sses_total[best_idx].item()

def get_sub_bucket_torch(X):
    best_bucket = (0, 0, float('inf'))
    for i in range(X.shape[1]):
        bucket = optimal_split_val_torch(X, i)
        if bucket[-1] < best_bucket[-1]:
            best_bucket = bucket
    return best_bucket

def get_ans_torch(X):
    buckets = []
    prototypes = []
    buckets.append(X)
    beg = 0
    end = 1
    ans = []
    
    for i in range(15):
        best_bucket = get_sub_bucket_torch(buckets[beg])

        X1 = buckets[beg][buckets[beg][:, best_bucket[0]] <= best_bucket[1]]
        X2 = buckets[beg][buckets[beg][:, best_bucket[0]] > best_bucket[1]]
        if X1.shape[0] == 0:
            buckets.append(X2)
            buckets.append(X2)
        elif X2.shape[0] == 0:
            buckets.append(X1)
            buckets.append(X1)
        else:
            buckets.append(X1)
            buckets.append(X2)
        end += 2
        beg += 1

        ans.append(best_bucket)

    for i in range(15, len(buckets)):
        if buckets[i].shape[0] > 0:
            prototypes.append(torch.mean(buckets[i], dim=0))
        else:
            prototypes.append(torch.zeros(X.shape[1]))

    return ans, prototypes

def create_maddness_supporters(train_data):
    D = 2
    S_all = torch.zeros((train_data.shape[1] // D, D, 15))
    T_all = torch.zeros((train_data.shape[1] // D, 15))
    LUT_all = torch.zeros((train_data.shape[1] // D, 16, D))
    
    for i in range(0, train_data.shape[1], D):
        X = train_data[:, i:i+D]
        ans, prototypes = get_ans_torch(X)
        for j in range(15):
            S_all[i // D, ans[j][0], j] = 1
            T_all[i // D, j] = ans[j][1]
        for j in range(16):
            LUT_all[i // D, j] = prototypes[j]

    return S_all, T_all, LUT_all