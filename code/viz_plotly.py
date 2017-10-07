import plotly.plotly as py
import plotly.graph_objs as go

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot


def interactive_scores_plot(scores, comp1, comp2, labels=None):
    """
    Plots scores with labels. Components are numbered starting at 1.

    Paramters
    ---------
    scores: numpy matrix of scores
    comp1: index of first component
    comp1: index of second component
    labels: list of points labels
    """
    if (comp1 <= 0) | (comp2 <= 0):
        raise ValueError('components should be positive')

    comp1 = comp1
    comp2 = comp2

    points_trace = go.Scatter(x=scores[:, comp1-1],
                              y=scores[:, comp2-1],
                              text=labels,
                              mode='markers')

    layout = go.Layout(title='',
                       hovermode='closest',
                       xaxis=dict(title='component %d' % comp1),
                       yaxis=dict(title='component %d' % comp2),
                       showlegend=False)

    fig = go.Figure(data=[points_trace], layout=layout)
    init_notebook_mode(connected=True)
    iplot(fig)
