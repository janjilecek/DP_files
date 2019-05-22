import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def clean_dataset(df):
    assert isinstance(df, pd.DataFrame), "df needs to be a pd.DataFrame"
    df.dropna(inplace=True)
    indices_to_keep = ~df.isin([np.nan, np.inf, -np.inf]).any(1)
    return df[indices_to_keep].astype(np.float64)

# checks integrity of an array
def _assert_all_finite(X):
    X = np.asanyarray(X)
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(X.sum())
            and not np.isfinite(X).all()):
        raise ValueError("Input contains NaN, infinity"
                         " or a value too large for %r." % X.dtype)


# annotate the graph with best value
def annot_max(x, y, ax=None, ee=[]):
    xmax = x[np.argmax(y)]
    ymax = max(y)
    text = "x={0}, y={1}".format(ee[xmax], str(ymax)[0:5])
    if not ax:
        ax = plt.gca()
    bbox_props = dict(boxstyle="round,pad=0.3", fc="w", ec="k", lw=0.72)
    from matplotlib.patches import Ellipse
    el = Ellipse((2, -1), 0.5, 0.5)

    arrowprops = dict(arrowstyle="simple",
                      facecolor='black',

                      patchB=el,
                      connectionstyle="angle3,angleA=0,angleB=-90")
    kw = dict(xycoords='data', textcoords="axes fraction",
              xytext=(0.45, 0.50),
              arrowprops=arrowprops, bbox=bbox_props, ha="right", va="top")
    ax.annotate(text, xy=(xmax, ymax), **kw)


def myplt(ests, trainresults, test_results, xlabel="x", ylabel="y"):
    from matplotlib.legend_handler import HandlerLine2D
    xn = range(len(ests))
    line1, = plt.plot(xn, trainresults, 'b', label="Train")
    line2, = plt.plot(xn, test_results, 'r', label="test")
    plt.legend(handler_map={line1: HandlerLine2D(numpoints=2)})
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(xn, ests)
    annot_max(xn, test_results, plt, ests)
    plt.savefig("fig_" + xlabel + "_" + ylabel + "_" + str(time.time()) + ".png")

    from matplotlib.backends.backend_pdf import PdfPages
    pp = PdfPages('multipage.pdf')
    plt.savefig(pp, format='pdf')
    pp.close()