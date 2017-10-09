import pandas as pd
import heapq

import matplotlib.pyplot as plt
import seaborn.apionly as sns

def top_loading_words(v, n, i2w):
    """
    Returns a list of the top words for a loading vector.

    Parameters
    ----------
    v: vector of loadings
    n: number of words to print
    iw: index to words
    """

    return heapq.nlargest(n, zip(abs(v), i2w))


def top_loading_words_df(V, i2w, n, include_coef=False):
    """
    Creates a data frame of the top words for each loading.
    Loadings are indexed starting at 1.

    Parameters
    ----------
    loading: the maxtrix of loading vectors as columns
    iw: index to word
    n: number of words for each loading
    include_coef: possibly include the loading coefficient

    Output
    ------
    pandas data frame whose
    """

    top_words_df = pd.DataFrame(index=range(1, n + 1),
                                columns=range(1, V.shape[1] + 1))

    w2i = {i2w[i]: i for i in range(len(i2w))}

    for k in range(V.shape[1]):
        top = top_loading_words(V[:, k], n, i2w)
        abs_load, words = zip(*top)
        if include_coef:
            words = ['%s (%.1E)' % (words[i], V[w2i[words[i]], k])
            for i in range(n)]

        top_words_df[k + 1] = words

    return top_words_df



def top_loading_components(v, i2w, n, comp_numer=None):
    """
    Shows the top n variables contributing to the loading

    Paramters
    ---------
    v: the laoding vector

    iw: list mapping index to word

    n: number of components to show
    """

    top_load = top_loading_words(v, n, i2w)
    top_words =[p[1] for p in top_load]  # list(zip(*top_load)[1])
    w2i = {i2w[i]: i for i in range(len(i2w))}
    load_vals = [v[w2i[w]] for w in top_words]

    # prevent unicode issues
    # top_words = [w.decode('utf-8') for w in list(top_words)]
    

    sns.barplot(y=top_words,
                x=load_vals,
                palette=['blue' if load_vals[i] > 0 else 'red' for i in range(n)])

    plt.title('loading %d' % comp_numer if comp_numer else '')
