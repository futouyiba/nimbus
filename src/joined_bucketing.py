import numba
import numpy as np

def get_splitted(bucket, dim, threshold):
    left_bucket = bucket[bucket[:, dim] < threshold]
    right_bucket = bucket[bucket[:, dim] >= threshold]
    return left_bucket, right_bucket

@numba.njit(fastmath=True, cache=True, parallel=False)
def _cumsse_dotprods(DotProds:np.ndarray):
    # TODO: can be optimized with numpy
    N, D = DotProds.shape
    cumsses = np.empty((N, D), DotProds.dtype)
    cumX_column = np.empty(D, DotProds.dtype)
    cumX2_column = np.empty(D, DotProds.dtype)
    for j in range(D):
        cumX_column[j] = DotProds[0, j]
        cumX2_column[j] = DotProds[0, j] * DotProds[0, j]
        cumsses[0, j] = 0  # no err in bucket with 1 element
    for i in range(1, N):
        one_over_count = 1.0 / (i + 1)
        for j in range(D):
            cumX_column[j] += DotProds[i, j]
            cumX2_column[j] += DotProds[i, j] * DotProds[i, j]
            meanX = cumX_column[j] * one_over_count
            cumsses[i, j] = cumX2_column[j] - (cumX_column[j] * meanX)
    return cumsses

def optimal_split_val(X, dim, B_slices:np.ndarray, X_orig=None):
    X_orig = X if X_orig is None else X_orig
    if X_orig.shape != X.shape:
        assert X_orig.shape == X.shape

    if X.shape[0] == 0:
        return dim, 0, 100000000000000000000

    N, _ = X.shape
    sort_idxs = np.argsort(X_orig[:, dim])
    dotProds = np.dot(X, B_slices.T)

    sortedProds = dotProds[sort_idxs]

    # cumulative SSE (sum of squared errors)
    sses_head = _cumsse_dotprods(sortedProds)
    sses_tail = _cumsse_dotprods(sortedProds[::-1])[::-1]
    sses = sses_head
    sses[:-1] += sses_tail[1:]
    sses = sses.sum(axis=1)

    best_idx = np.argmin(sses)
    next_idx = min(N - 1, best_idx + 1)
    col = X[:, dim]
    best_val = (col[sort_idxs[best_idx]] + col[sort_idxs[next_idx]]) / 2

    return dim, best_val, sses[best_idx]

def split_bucket_optimal(bucket, weight_slices):
    best_sse = float('inf')
    best_dim = None
    best_threshold = None
    best_split = None

    if bucket.size == 0:
        return best_dim, best_threshold, (None, None), best_sse

    for dim in range(bucket.shape[1]):
        dim, threshold, sse = optimal_split_val(bucket, dim, weight_slices)

        if sse < best_sse:
            best_sse = sse
            best_dim = dim
            best_threshold = threshold

    best_split = get_splitted(bucket, best_dim, best_threshold)
    
    return best_dim, best_threshold, best_split, best_sse

def splitFunc(bucket_id, buckets, dims, thresholds, weight_slices, max_depth):
    print(f"start split function with bucket_id:{bucket_id}, max_depth:{max_depth}")
    if max_depth <= 0 or bucket_id >= len(buckets):
        return
    bucket = buckets[bucket_id]
    if bucket is None or bucket.size == 0:
        return

    dim, threshold, (left_bucket, right_bucket), _ = split_bucket_optimal(bucket, weight_slices)
    dims[bucket_id] = dim
    thresholds[bucket_id] = threshold

    left_index = 2 * bucket_id + 1
    right_index = 2 * bucket_id + 2

    if left_index < len(buckets):
        buckets[left_index] = left_bucket
    if right_index < len(buckets):
        buckets[right_index] = right_bucket
    
    print(f"bucket splitted with dim:{dim}, threshold:{threshold}, from bucket_id:{bucket_id}'s shape {bucket.shape}, to bucket id {left_index} with length {left_bucket.shape} & bucket id {right_index} with length {right_bucket.shape}")

    splitFunc(left_index, buckets, dims, thresholds, weight_slices, max_depth - 1)
    splitFunc(right_index, buckets, dims, thresholds, weight_slices, max_depth - 1)

def learnMaddnessJoined(X:np.ndarray, B:np.ndarray, C:int, H:int):
    N, D = X.shape
    M, _ = B.shape
    assert D == B.shape[1]
    assert D % C == 0
    d = D // C
    X_reshaped = X.reshape(N, C, d)
    B_reshaped = B.reshape(M, C, d)
    K = 2 ** H  # number of leaf buckets
    numAllBucketsPerBlock = 2 * K - 1 # split nodes and leaf buckets
    numSplitNodesPerBlock = K - 1
    selectionMatrix = np.zeros((C, numSplitNodesPerBlock, d), dtype=np.float32)
    thresholds = np.zeros((C, numSplitNodesPerBlock), dtype=np.float32)
    dims = np.zeros((C, numSplitNodesPerBlock), dtype=np.int64) # int64 for indexing
    luts = np.zeros((C, K, M), dtype=np.float32)
    luts_simple_avg = np.zeros((C, K, M), dtype=np.float32)
    for c in range(C):
        bucketsThisBlock = [None] * numAllBucketsPerBlock
        bucketsThisBlock[0] = X_reshaped[:, c, :]
        dimsThisBlock = dims[c]
        thresholdsThisBlock = thresholds[c]
        weight_slices = B_reshaped[:, c, :]
        splitFunc(0, bucketsThisBlock, dimsThisBlock, thresholdsThisBlock, weight_slices, H)

        # fill selection matrix
        for i in range(numSplitNodesPerBlock):
            selectionMatrix[c, i, dimsThisBlock[i]] = 1

        # ridge regression fit & lut calculation
        leafBuckets = bucketsThisBlock[numSplitNodesPerBlock:]
        for k, bucket in enumerate(leafBuckets):
            if bucket is None:
                continue
            Xk = bucket # shape (Nk, d)
            Bk = B_reshaped[:, c, :] # shape (M, d)
            # XkTXk = np.dot(Xk.T, Xk)
            # XkTXk_inv = np.linalg.inv(XkTXk + 0.01 * np.eye(d))
            # XkTXk_inv_XkT = np.dot(XkTXk_inv, Xk.T)
            # Bk = Bk.reshape(-1, 1)
            # luts[c, k] = np.dot(XkTXk_inv_XkT, Bk).reshape(-1)
            # simple luts avg calculation
            dotProds = np.dot(Xk, Bk.T) # shape (Nk, M)
            luts_simple_avg[c, k] = dotProds.mean(axis=0)
            # if debug mode, calculate NMSE between luts and simple luts avg
            # NMSE = np.mean((luts[c, k] - luts_simple_avg[c, k]) ** 2) / np.mean(luts_simple_avg[c, k] ** 2)
            # print (f"NMSE between luts and simple luts avg: {NMSE} (c:{c}, k:{k})")

    # return selectionMatrix, thresholds, dims, luts
    # return selectionMatrix, thresholds, dims, luts, luts_simple_avg
    return selectionMatrix, thresholds, dims, luts_simple_avg