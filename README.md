# Convolution neuron network from scratch
This notebook implement a mini deep learning frame-work in the style of pytorch. Everything here from RELU, Batch Norm, Softmax, ...  are implemented from scratch. 

<img src="yes-no.png" style="width:600px;height:300px;">


The purpose of this notebook is to encourage understanding of deep learning, everything under the hood how these stuffs works (which is not scary at all) and the frameworks that implement them. A little bit of programming skill and linnear algebra is sufficient to work your way from bottom up to implement convolution neural networks. I also include a graident check function that helps validate back-probagation implement.

The list of layers and activation that were implemented in this repo:

2D Convolution (http://lushuangning.oss-cn-beijing.aliyuncs.com/CNN%E5%AD%A6%E4%B9%A0%E7%B3%BB%E5%88%97/Gradient-Based_Learning_Applied_to_Document_Recognition.pdf)
Pooling (max and average) 
Batch normalization (https://arxiv.org/abs/1502.03167)
RELU-Rectified Linear Unit (https://www.cs.toronto.edu/~fritz/absps/reluICML.pdf) 
Linear
Softmax
Adam optimization (https://arxiv.org/abs/1412.6980)
Cross entropy loss

MNIST and CIFAR10 for evaluation.

The network declaration and using is similar to that of pytorch. 
```python
class CNN:
    """
    Typical pytorch style network declaration.
    """
    def __init__(self, in_shape, out_size):

        in_size = in_shape[0]
        in_chn = in_shape[-1]

        conv1_channel = 12
        self.convo1 = Conv2D(in_chn=in_chn, out_chn=conv1_channel, kernel_size=3, in_shape=in_shape, padding=1, stride=2, bias=False)

        c1 = (in_size-3 + 2*1)//2 + 1
        output_shape = (c1,c1,conv1_channel)

        self.batchnorm1 = BatchNorm(output_shape)

        conv2_channel = 2*conv1_channel
        self.convo2 = Conv2D(in_chn=conv1_channel, out_chn=conv2_channel, kernel_size=3, in_shape=output_shape, padding=1, stride=2, bias=False)

        c2 = (c1-3 + 2*1)//2 + 1
        output_shape = (c2,c2,conv2_channel)

        self.batchnorm2 = BatchNorm(output_shape)

        self.flatten = Flatten()


        linear_in = np.prod(output_shape)

        self.linear_softmax = Linear_SoftMax(linear_in, out_size)

        # only layers with trainable weights here, which are used in optimization/gradient update.
        self.layers = {'convo1': self.convo1, 'batch_norm1': self.batchnorm1, 'convo2': self.convo2, 
                       "batch_norm2": self.batchnorm2, 'linear_softmax': self.linear_softmax}

    def forward(self, X):

        X = self.convo1.forward(X)
        X = self.batchnorm1.forward(X)

        X = self.convo2.forward(X)
        X = self.batchnorm2.forward(X)

        X = self.flatten.forward(X)
        X = self.linear_softmax.forward(X)

        return X

    def backward(self, dZ):

        dZ = self.linear_softmax.backward(dZ)
        dZ = self.flatten.backward(dZ)

        dZ = self.batchnorm2.backward(dZ)
        dZ = self.convo2.backward(dZ)

        dZ = self.batchnorm1.backward(dZ)
        dZ = self.convo1.backward(dZ)

        return dZ

    def set_weights(self, weight_list):
        for k, (W,b) in weight_list.items():
            self.layers[k].W = W
            self.layers[k].b = b

    def get_weights(self):
        return {k:(layer.W, layer.b) for k,layer in self.layers.items()}

    def get_dweights(self):
        return {k:(layer.dW, layer.db) for k,layer in self.layers.items()}
```
