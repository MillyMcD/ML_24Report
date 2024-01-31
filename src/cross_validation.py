from tqdm import tqdm
from data_loading import get_train_and_test_set
from metrics import *

def perform_cross_validation(dataset,architecture,splits,dependent_column,mode='classification',normalise=True,
                             use_loading_bar:bool=False):
    """
    Perform cross validation
    """
    metrics = get_classification_metrics() if mode == 'classification' else get_regression_metrics()

    if use_loading_bar:
        splits = tqdm(splits)

    for split in splits:
        (trn_x,trn_y),(tst_x,tst_y) = get_train_and_test_set(dataset,split,dependent_column=dependent_column,
                                                             normalise=normalise)
        model = architecture()
        model.fit(trn_x,trn_y)
        preds = model.predict(tst_x)

        for _,metric in metrics.items():
            metric.update(tst_y,preds)

    reports = {'arch':architecture.__name__,'normalise':normalise,'num_features':dataset.shape[1]}
    for name,metric in metrics.items():
        if name == 'residual':
            reports['residual']=metric.values
        else:
            mean, std = metric.summarise()
            reports[f'{name} mean'] = mean
            reports[f'{name} std'] = std
    return reports