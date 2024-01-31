import numpy as np
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

def create_train_test_splits(df,n_splits=1,test_size=0.2,stratified=False,dependent_column='Explicit'):
    """
    Create train test split for a dataframe
    """
    unique_ids = df['Track URI']

    train_test_splits = []
    #use stratified for classification
    if stratified and dependent_column in df.columns:
        splitter = StratifiedShuffleSplit(n_splits=n_splits,test_size=test_size)
        labels   = df[dependent_column]
        
        for split in splitter.split(unique_ids,labels):
            train = sorted(unique_ids[split[0]])
            test  = sorted(unique_ids[split[1]])
            train_test_splits.append({'train':train,'test':test})
    #for everything else
    else:
        splitter = ShuffleSplit(n_splits=n_splits,test_size=test_size)
        
        for split in splitter.split(unique_ids):
            train = sorted(unique_ids[split[0]])
            test  = sorted(unique_ids[split[1]])
            train_test_splits.append({'train':train,'test':test})

    return train_test_splits

def normalise_data(df,dependent_column=None,norm_stats=None):
    """
    Normalise all non-object columns via mean/standard deviation
    """
    if dependent_column is None:
        dependent_column = []
    else:
        dependent_column = [dependent_column]
    
    non_obj_cols = [i for i in df.select_dtypes([np.number,bool]).columns if i not in dependent_column]

    if norm_stats is None:
        norm_stats = {'mean':{},'std':{}}
        
    for i in non_obj_cols:
        if i not in norm_stats['mean']:
            mean = df[i].mean()
            std  = df[i].std()+1.0
            norm_stats['mean'][i]=mean
            norm_stats['std'][i]=std
        else:
            mean = norm_stats['mean'][i]
            std  = norm_stats['std'][i]
        
        df.loc[:,i] = (df[i] - mean)/std
    return df, norm_stats

def get_train_and_test_set(df,split,dependent_column=None,normalise:bool=False):
    """
    Get train and test set
    """
    train = df.loc[df['Track URI'].isin(split['train'])]
    if normalise:
        train,norm_stats = normalise_data(df=train,dependent_column=dependent_column)

    test  = df.loc[df['Track URI'].isin(split['test'])]
    if normalise:
        test,_ = normalise_data(df=test,dependent_column=dependent_column,norm_stats=norm_stats)

    train = train.drop(columns=['Track URI'])
    test  = test.drop(columns=['Track URI'])

    if dependent_column is not None:
        train_y = train[dependent_column].values
        test_y  = test[dependent_column].values
        
        train   = train.drop(columns=[dependent_column])
        test    = test.drop(columns=[dependent_column])
        train_x = train.values
        test_x  = test.values

        return (train_x, train_y), (test_x, test_y)
    else:
        train_x = train.values
        test_x  = test.values
        return (train_x, test_x)

def get_unsupervised_data(df,normalise:bool=False):
    """
    For unsupervised modelling
    """
    if normalise:
        df,norm_stats = normalise_data(df=df)
    df = df.drop(columns=['Track URI'])
    return df
    