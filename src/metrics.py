import numpy as np

from sklearn.metrics import *

class MetricTracker:
    """
    For tracking metrics over cross validation runs
    """
    def __init__(self,metric,args):
        self.metric = metric
        self.args   = args
        self.values = []
        self.mean = 0
        self.std  = 0

    def update(self,y_true,y_pred):
        value = self.metric(y_true,y_pred,**self.args)

        if self.metric != np.subtract:
            self.values.append(value)
       
            self.mean = np.mean(self.values,axis=0)
            self.std  = np.std(self.values,axis=0)
        else:
            self.values.extend(value)

    def summarise(self):
        return self.mean, self.std


def get_classification_metrics():
    """ get classification trackers """
    metrics = {'f1':MetricTracker(f1_score,args={'average':'macro'}),
               'acc':MetricTracker(accuracy_score,{}),
               'cm':MetricTracker(confusion_matrix,{'normalize':'true'})}
    return metrics

def get_regression_metrics():
    """get regression trackers """
    metrics = {'rmse':MetricTracker(mean_squared_error,{'squared':False}),
               'r2':MetricTracker(r2_score,{}),
               'residual':MetricTracker(np.subtract,{})}
    return metrics

