from scipy.sparse import csr_matrix, dok_matrix
import numpy as np

def calc_epmi(counts, cds=1.0):
    """
    Calculates e^PMI; PMI without the log().
    
    Parameters
    counts: matrix of word-word counts
    
    cds: context distribution smooting
    """
    sum_w = np.array(counts.sum(axis=1))[:, 0]
    sum_c = np.array(counts.sum(axis=0))[0, :]

    if cds != 1:
        sum_c = sum_c ** cds

    sum_total = sum_c.sum()
    sum_w = np.reciprocal(sum_w)
    sum_c = np.reciprocal(sum_c)

    epmi = csr_matrix(counts)
    epmi = multiply_by_rows(epmi, sum_w)
    epmi = multiply_by_columns(epmi, sum_c)
    epmi = epmi * sum_total
    return epmi

def multiply_by_rows(matrix, row_coefs):
    normalizer = dok_matrix((len(row_coefs), len(row_coefs)))
    normalizer.setdiag(row_coefs)
    return normalizer.tocsr().dot(matrix)


def multiply_by_columns(matrix, col_coefs):
    normalizer = dok_matrix((len(col_coefs), len(col_coefs)))
    normalizer.setdiag(col_coefs)
    return matrix.dot(normalizer.tocsr())

def calc_ppmi(counts, neg=1.0, cds=1.0):
    """
    Calcluates the PPMI from an e^pmi matrix
    
    Parameters
    ----------
    epmi: e^pmi marix
    neg:
    
    """
    m = calc_epmi(counts, cds)
    
    m.data -= np.log(neg)
    m.data[m.data < 0] = 0
    m.eliminate_zeros()
    return m