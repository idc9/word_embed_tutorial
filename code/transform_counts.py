import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import norm


def normalize_rows(counts):
    """
    Normalize rows by their norms
    """
    return diags(1.0/norm(counts, axis=1)) * counts 


def nz_elementwise_transform(counts, f):
    """
    Applies the function f to each non-zero entry of counts. 

    nz_elementwise_transform(counts, lambda x: np.log(1 + x))

    Parameters
    ----------
    counts:
    f: function to apply to nonzero entries -- must be vectorized
    """
    counts_trans = counts.copy()
    counts_trans.data = f(counts.data)
    return counts_trans


def remove_zero_count_words(counts, i2w):
    """
    Removes rows with zero counts
    """

    # remove rows/columns with zeros
    # to_keep = np.logical_and((counts.sum(0) != 0).A1,
    #                          (counts.sum(1) != 0).A1)

    # keep words with nonzero row counts
    to_keep = (counts.sum(axis=0) != 0).A1

    counts = counts[to_keep, :]
    counts = counts[:, to_keep]

    # update iw, wi
    i2w = np.array(i2w)[to_keep].tolist()
    w2i = {i2w[i]: i for i in range(len(i2w))}
    
    return counts, w2i, i2w