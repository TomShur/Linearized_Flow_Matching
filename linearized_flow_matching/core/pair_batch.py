import torch
from scipy.optimize import linear_sum_assignment

@torch.no_grad()
def pair_batch(x0, x1, p=1):
    batch_size = x0.shape[0]
    x0_flat = x0.view(batch_size, -1) # flatten
    x1_flat = x1.view(batch_size, -1)
    cost_matrix = torch.cdist(x0_flat, x1_flat, p=p).cpu().numpy() # calculate bipirtite matching pairing where distance is L2

    # linear_sum_assignment finds the optimal non-overlapping pairs
    # row_ind corresponds to x0 indices, col_ind corresponds to x1 indices
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # create a new x0 tensor where the noise is sorted to match the order of x1
    # sort the rows of x0 based on where the algorithm told them to go (col_ind)
    sorted_x0 = torch.zeros_like(x0)
    for i in range(batch_size):
        sorted_x0[col_ind[i]] = x0[row_ind[i]]

    return sorted_x0