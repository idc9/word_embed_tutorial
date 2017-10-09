import numpy as np
import matplotlib.pyplot as plt
import seaborn.apionly as sns
import heapq

def top_counts_bar_plot(counts, i2w, N=10, title='', figsize=[10, 10]):
    """
    Makes a bar plot of the largest counts.

    Parameters
    ----------

    counts: numpy array of counts
    lables: numpy array of lables
    num: how many counts to display
    title: title of the plot
    figsize:
    """

    largest = heapq.nlargest(N, zip(counts, i2w))
    top_counts, top_words = zip(*largest)

    # prevent unicode issues
    # top_words = [w.decode('utf-8') for w in list(top_words)]

    plt.figure(figsize=figsize)
    sns.barplot(y=top_words, x=list(top_counts), color='grey')

    plt.xlabel('counts')
    plt.title('%d most frequently occurring words' % N)

    plt.xlabel('counts')
    plt.title(title)
    
    
def co_counts_intersection(co_counts, word1, word2, w2i, i2w, threshold=0, just_words=False):
    """
    Finds the set of words that co-occur with two given words.
    
    Parameters
    ----------
    co_counts: matrix of word co-occurence counts
    
    word1, word2: two words whose intersection to find
    
    i2w, w2i: map indices to words and words to indices
    
    threshold: miniumum number of co occurences
    
    just_words: whether or not to return the values
    """

    co_counts_1 = co_counts[w2i[word1], :].todense().A1
    non_zero_1 = set(np.where(co_counts_1 > threshold)[0])

    co_counts_2 = co_counts[w2i[word2], :].todense().A1
    non_zero_2 = set(np.where(co_counts_2 > threshold)[0])

    # find intersection
    non_zero_both = list(non_zero_1.intersection(non_zero_2))
    words_both = [i2w[w] for w in non_zero_both]


    if just_words:
        return words_both

    else:
        c1 = [co_counts_1[w2i[w]] for w in words_both]
        c2 = [co_counts_2[w2i[w]] for w in words_both]

        return words_both, c1, c2