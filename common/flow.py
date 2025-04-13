import numpy as np
from numba import njit

# D8 direction codes (powers of 2) and their (dy, dx)
d8_codes = np.array([1, 2, 4, 8, 16, 32, 64, 128], dtype=np.uint8)
d8_offsets = np.array([
    ( 0,  1),  # E
    ( 1,  1),  # SE
    ( 1,  0),  # S
    ( 1, -1),  # SW
    ( 0, -1),  # W
    (-1, -1),  # NW
    (-1,  0),  # N
    (-1,  1)   # NE
], dtype=np.int8)


@njit
def compute_flow_direction(dtm):
    rows, cols = dtm.shape
    flow_dir = np.zeros((rows, cols), dtype=np.uint8)

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            min_elev = dtm[i, j]
            min_dir = 0  # default: sink

            for k in range(8):
                di, dj = d8_offsets[k]
                ni, nj = i + di, j + dj
                elev = dtm[ni, nj]

                if elev < min_elev:
                    min_elev = elev
                    min_dir = d8_codes[k]

            flow_dir[i, j] = min_dir

    return flow_dir


@njit
def compute_flow_accumulation(flow_dir):
    rows, cols = flow_dir.shape
    acc = np.ones((rows, cols), dtype=np.int32)  # each cell contributes at least itself
    in_degree = np.zeros((rows, cols), dtype=np.int32)
    receivers = np.full((rows, cols, 2), -1, dtype=np.int32)

    # 1. Build the flow graph and in-degrees
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            dir_code = flow_dir[i, j]
            for k in range(8):
                if dir_code == d8_codes[k]:
                    di, dj = d8_offsets[k]
                    ni, nj = i + di, j + dj
                    receivers[i, j, 0] = ni
                    receivers[i, j, 1] = nj
                    in_degree[ni, nj] += 1
                    break

    # 2. Topological sort queue
    queue_i = np.empty(rows * cols, dtype=np.int32)
    queue_j = np.empty(rows * cols, dtype=np.int32)
    head = 0
    tail = 0

    for i in range(rows):
        for j in range(cols):
            if in_degree[i, j] == 0:
                queue_i[tail] = i
                queue_j[tail] = j
                tail += 1

    # 3. Propagate accumulation
    while head < tail:
        i = queue_i[head]
        j = queue_j[head]
        head += 1

        ni, nj = receivers[i, j]
        if ni != -1:
            acc[ni, nj] += acc[i, j]
            in_degree[ni, nj] -= 1
            if in_degree[ni, nj] == 0:
                queue_i[tail] = ni
                queue_j[tail] = nj
                tail += 1

    return acc
