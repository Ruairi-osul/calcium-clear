import numpy as np
import sklearn.metrics


def auc(arr, to_1=True):
    if to_1:
        x = np.linspace(0, 1, len(arr))
    else:
        x = np.arange(len(arr))
    try:
        return sklearn.metrics.auc(x, arr)
    except ValueError:
        return np.nan
