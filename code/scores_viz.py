import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

import collections
import seaborn.apionly as sns
from webcolors import name_to_rgb


def scores_plot(scores,
                start=1,
                n_comp=3,
                classes=None,
                cvals=None,
                cmap=None,
                ind2ephasize=None,
                colors=None,
                alpha=.5,
                title='',
                comp_names=None):
    """
    Makes the scores plot i.e. an upper triangular grid of plots where the
    main diagonal is a histogram of each score component and the off diagonal
    entries i,j are a scatter plot of the ith score vs. the jth score.

    Parameters
    ----------
    scores: matrix whose columns are the scores vectors (e.g. n rows where
    n = numer of data points)

    start: which score to start at

    n_comp: how many scores to plot

    classes: (optional) list of class labels, will call multiclass_scores_plot

    cvals: (optional) values used for color mapping
    cmap: (optional) color mapping

    ind2ephasize: indices of points to emphasize

    colors: (optional) colors for points on the scatter plot

    alpha: (optional) alpha for points or for de-emphasized points
    title: (optional) title

    comp_names: (optional) names for the components
    """
    if classes is not None:
        multiclass_scores_plot(scores,
                               classes,
                               start,
                               n_comp,
                               title,
                               comp_names)

        return

    if (start < 1) | ((start + n_comp) > scores.shape[1] + 1):
        raise ValueError('start of end out of range')

    if type(scores) == pd.core.frame.DataFrame:
        comp_names = scores.columns.tolist()
        scores = scores.as_matrix()

    if comp_names is None or len(comp_names) < n_comp:
        comp_names = ['comp %d' % (j+1) for j in range(scores.shape[1])]

    if colors is None:
        colors = ['grey']*scores.shape[0]

    start -= 1  # zero indexing

    plt.figure(figsize=[5 * n_comp, 5 * n_comp])
    plt.title(title)
    p = 1
    for i in range(start, start + n_comp):
        for j in range(start, start + n_comp):
            if i == j:
                plt.subplot(n_comp, n_comp, p)

                plot_jitter_hist(scores[:, i])

                plt.ylabel(comp_names[i])
            elif i < j:

                plt.subplot(n_comp, n_comp, p)

                # color by values
                if cvals is not None:
                    plt.scatter(scores[:, j], scores[:, i], c=cvals, cmap=cmap)
                    plt.colorbar()

                elif ind2ephasize is not None:
                    # plot ephasized points on top of deempasized points
                    plt.scatter(np.delete(scores, ind2ephasize, axis=0)[:, j],
                                np.delete(scores, ind2ephasize, axis=0)[:, i],
                                alpha=alpha, color='grey')

                    plt.scatter(scores[ind2ephasize, j],
                                scores[ind2ephasize, i],
                                alpha=1, color='red', zorder=2)

                else:
                    plt.scatter(scores[:, j], scores[:, i],
                                alpha=alpha, color=colors)

                plt.xlabel(comp_names[j])
                plt.ylabel(comp_names[i])

                if i == 0 and j == 1:
                    plt.title(title)

            p += 1


def plot_jitter_hist(y, n_bins=10, xlabel='', title=''):
    """
    Jitter plot histogram
    """
    n = len(y)

    # plot the histogram of the points
    bins = np.linspace(min(y), max(y), n_bins)
    hist = plt.hist(y, bins=bins, alpha=.5, color='grey')
    counts, bins = hist[0], hist[1]

    jitters = get_jitter(N=n, base=max(counts) * .1, width=max(counts) * .01)
    plt.scatter(y, jitters,
                color='black', alpha=.7, s=1, zorder=2)

    plt.xlim([min(bins), max(bins)])
    plt.ylim([0, max(counts)])

    plt.xlabel(xlabel)
    plt.title(title)

    # # points where we evaluate the pdf of the kde
    # p = np.linspace(min(bins), max(bins), 1000)
    #
    # # get guassian KDE of individual classes
    # y_kde = gaussian_kde(y0, bw_method='scott').pdf(p)
    #
    # # plot the class specific kdf pdfs
    # plt.plot(p, y_kde, color='blue')


def multiclass_scores_plot(scores,
                           classes,
                           start=1,
                           n_comp=3,
                           title='',
                           comp_names=None):
    """
    Makes a scores plot where points are colored by classes.

    Parameters
    ----------
    scores: matrix whose columns are the scores vectors (e.g. n rows where
    n = numer of data points)

    classes: list class labels for each point

    start: which score to start at

    n_comp: how many scores to plot

    colors: (optional) colors for points on the scatter plot

    title: (optional) title

    comp_names: (optional) names for the components
    """

    if (start < 1) | (start + n_comp) > scores.shape[1] + 1:
        raise ValueError('start of end out of range')

    if type(scores) == pd.core.frame.DataFrame:
        comp_names = scores.columns.tolist()
        scores = scores.as_matrix()

    if comp_names is None or len(comp_names) < n_comp:
        comp_names = ['comp %d' % (j+1) for j in range(scores.shape[1])]

    class2col = class_to_color(classes, class_alphas=True)
    colors = [class2col[classes[i]] for i in range(len(classes))]

    # make plots
    plt.figure(figsize=[5 * n_comp, 5 * n_comp])
    plt.title(title)
    p = 1
    start -= 1  # zero indexing
    for i in range(start, start + n_comp):
        for j in range(start, start + n_comp):
            if i == j:

                plt.subplot(n_comp, n_comp, p)

                multiclass_hist(scores[:, i],
                                classes=classes,
                                class2col=class2col,
                                legend=(i == j ==0))

                plt.ylabel(comp_names[i])
            elif i < j:

                plt.subplot(n_comp, n_comp, p)

                plt.scatter(scores[:, j], scores[:, i], color=colors)

                plt.xlabel(comp_names[j])
                plt.ylabel(comp_names[i])

                if i == 0 and j == 1:
                    plt.title(title)

            p += 1


def multiclass_hist(x, classes, class2col=None, legend=True, xlabel=''):
    """
    Plots a histogram with a KDE for each subclass.

    Parameters
    ----------
    x: list of points
    classes: list class labels for each point
    class2col: optional color map from class labels to colors
    legend: whether or not to plot the legend
    xlabel:
    """

    class_labels = list(set(classes))

    if not class2col:
        class2col = class_to_color(classes, class_alphas=True)

    colors = [class2col[classes[i]] for i in range(len(classes))]

    # get histogram of all points
    hist = plt.hist(x, alpha=.3, color='black')
    counts, bins = hist[0], hist[1]

    # show each individual point using a jitter plot
    # jitter_y = np.percentile(counts, .25)
    jitter_y = max(counts) * .1
    plt.scatter(x, np.random.uniform(0, 1, len(x)) + jitter_y,
                color=colors, zorder=2)

    plt.xlim([min(bins), max(bins)])
    plt.ylim([0, max(counts)])

    # points where we evaluate the pdf of the kde
    pts = np.linspace(min(bins), max(bins), 1000)

    # grab points in each class
    for lab in class_labels:
        sub = [x[i] for i in range(len(x)) if classes[i] == lab]

        # kde evaluated at pts
        sub_kde = gaussian_kde(sub, bw_method='scott').pdf(pts)

        rescale = (max(counts) + 0.0) / max(sub_kde) * .75
        sub_kde = np.array(sub_kde) * rescale

        # plot the class specific kdf pdfs
        plt.plot(pts, sub_kde, color=class2col[lab], label=lab)

    if legend:
        plt.legend(loc='upper right')


def get_jitter(N, base, width):
    return base + np.random.uniform(low=-width, high=width, size=N)


def filter_scores(U, comp1, comp2=None, comp1_min=-np.inf, comp1_max=np.inf,
                  comp2_min=-np.inf, comp2_max=np.inf, index=True):
    """
    Filters matrices by compoents for scores plots.
Components are numbered starting at 1.
    Paramters
    ---------
    U: scores matrix (numpy matrix)
    comp1: first component to filter
    comp2: second component to filter (optional)
    comp1_min: thresholds

    Output
    ------
    indices
    """
    if comp1 <= 0:
        raise ValueError('comp1 should be positive')

    comp1 = comp1 - 1

    indices = (comp1_min <= U[:, comp1]) & (U[:, comp1] <= comp1_max)

    if comp2:
        indices = indices & filter_scores(U, comp1=comp2,
                                          comp1_min=comp2_min,
                                          comp1_max=comp2_max,
                                          index=False)

    if index:
        return np.where(indices)[0]
    else:
        return indices



def class_to_color(classes, class_alphas=False):
    """
    Returns a dictionary mapping class label to color
    """
    class_labels = list(set(classes))
    pallette = sns.color_palette("Set2", len(class_labels))
    class2col = {class_labels[k]: pallette[k]
                 for k in range(len(class_labels))}

    # possibly add alphas
    if class_alphas:
        class2alpha = class_to_alpha(classes)
        class2col = {k: (class2col[k][0], class2col[k][1], class2col[k][2],
                         class2alpha[k]) for k in class2col.keys()}

    return class2col


def class_to_alpha(classes):
    """
    Returns a dictionary mapping class label to alpha value so that
    large classes recieve smaller alpha.
    """
    class_labels = list(set(classes))

    class2count = dict(collections.Counter(classes))
    class2prop = {k: 1 - (class2count[k]+0.0)/sum(class2count.values())
                  for k in class2count.keys()}

    return class2prop


def highlight_to_col(n, emph_ind,
                     color='blue', emph_color='red',
                     alpha=.1, emph_alpha=1):
    """
    Color vector which highlights selected indices.

    Parameters
    ----------
    n: length of color vector
    emph_ind: indices to highlight
    color: base color for each point
    emph_color: color for emphasized points
    alpha: base alpha for each point
    emph_alpha: alpha value for each emphasized points

    Output
    ------
    list of colors
    """

    rgb = rgb_255_to_decimal(name_to_rgb(color))
    emp_rgb = rgb_255_to_decimal(name_to_rgb(emph_color))

    return[(emp_rgb[0], emp_rgb[1], emp_rgb[2], emph_alpha)if i in emph_ind
           else (rgb[0], rgb[1], rgb[2], alpha)
           for i in range(n)]


def rgb_255_to_decimal(rgb):
    """
    Converts rbg from 255 scale to 0-1 scale

    >>> rgb_255_to_decimal((255, 255, 0))
    """
    return (rgb[0]/255.0, rgb[1]/255.0, rgb[2]/255.0)

