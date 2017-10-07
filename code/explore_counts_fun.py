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
    top_words = [w.decode('utf-8') for w in list(top_words)]

    plt.figure(figsize=figsize)
    sns.barplot(y=top_words, x=list(top_counts), color='grey')

    plt.xlabel('counts')
    plt.title('%d most frequently occurring words' % N)

    plt.xlabel('counts')
    plt.title(title)