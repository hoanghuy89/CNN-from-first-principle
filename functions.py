import numpy as np


def train_test_split(X,y,ratio=0.2,shuffle = True):

    m = X.shape[0]
    if shuffle==True:
        index = np.arange(m)
        np.random.shuffle(index)
        X = X[index]
        y = y[index]
    
    split = np.round((1-ratio)*m).astype(int)
    
    X_train = X[:split]
    y_train = y[:split]
    
    X_test = X[split:]
    y_test = y[split:]
    
    return X_train, y_train,  X_test, y_test

def one_hot(y, nclass):
    n = y.size
    Y = np.zeros((n,nclass))
    Y[np.arange(n),y] = 1 

    return Y

def he_normal(out_shape, fan_in):
    return np.random.randn(*out_shape) * np.sqrt(2./fan_in)
    

def pickle_compress(file_path, data=None, operation='load'):
    import bz2
    import pickle
    import _pickle as cPickle
    
    a = None
    if operation != 'load':
        with bz2.BZ2File(file_path, 'wb') as f:
            cPickle.dump(data, f)
    else:
        with bz2.BZ2File(file_path, 'rb') as f:
            a = cPickle.load(f)
    return a

def plot_confusion_matrix(y_actu, y_pred, title='Confusion matrix'):
    cmap=plt.cm.gray_r
    import matplotlib.pyplot as plt
    import pandas as pd

    df_confusion = pd.crosstab(y_actu, y_pred.reshape(y_pred.shape[0],), rownames=['Actual'], colnames=['Predicted'], margins=True)
    
    df_conf_norm = df_confusion / df_confusion.sum(axis=1)
    
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(df_confusion.columns))
    plt.xticks(tick_marks, df_confusion.columns, rotation=45)
    plt.yticks(tick_marks, df_confusion.index)
    plt.ylabel(df_confusion.index.name)
    plt.xlabel(df_confusion.columns.name)
    
