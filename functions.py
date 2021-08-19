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
    


"""
Implements the pooling layer with max or average pooling.
Max pooling put a filter over input in sliding window fasion, take the maximum cell in the filter
 and put it into the output
This implementation mostly taken from coursera deeplearning.ai course
"""

"""
Implements the forward pass of the pooling layer

Arguments:
A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
hparameters -- python dictionary containing "f" and "stride"
mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

Returns:
A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
"""
#        (array([[[[ 0.94403741, -0.        ],
#          [-0.        ,  0.35572819],
#          [ 0.33408715,  0.3541151 ]],

#         [[-0.        ,  1.9870983 ],
#          [-0.        , -0.        ],
#          [ 1.65719765, -0.        ]],

#         [[-0.        , -0.        ],
#          [ 1.50316071, -0.        ],
#          [-0.        , -0.        ]]],


#        [[[ 0.31046488, -0.        ],
#          [ 0.06424836,  0.90388799],
#          [ 0.36075398,  0.09863601]],

#         [[-0.        ,  0.0473638 ],
#          [ 0.02029675,  0.15701879],
#          [ 0.89011194,  0.33130962]],

#         [[ 0.37800902,  0.24378408],
#          [-0.        , -0.        ],
#          [ 0.05497921,  0.33165276]]]]),
# (2, 3, 3, 2))

import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = int((H + 2 * padding - field_height) / stride + 1)
    out_width = int((W + 2 * padding - field_width) / stride + 1)

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return (k.astype(int), i.astype(int), j.astype(int))


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """ An implementation of im2col based on some fancy indexing """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding, stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded
    return x_padded[:, :, padding:-padding, padding:-padding]