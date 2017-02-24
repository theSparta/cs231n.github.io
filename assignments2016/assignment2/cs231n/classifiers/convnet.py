import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class MyThreeLayerConvNet(object):
  """
  A three-layer convolutional network with the following architecture:

  conv - norm - relu - 2x2 max pool - affine - norm - relu - affine - softmax

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
               hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
               dtype=np.float32):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn_params = []

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #
    # Store weights and biases for the convolutional layer using the keys 'W1' #
    # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
    # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
    # of the output affine layer.                                              #
    ############################################################################
    # pass
    C, H, W = input_dim
    dim1 = (num_filters, C, filter_size, filter_size)
    self.params['W1'] = weight_scale * np.random.randn(*dim1)
    self.params['b1'] = np.zeros(num_filters)

    dim1 = H//2 * W//2 * num_filters
    self.params['W2'] = weight_scale * np.random.randn(dim1, hidden_dim)
    self.params['b2'] = np.zeros(hidden_dim)
    self.params['W3'] = weight_scale * np.random.randn(hidden_dim, num_classes)
    self.params['b3'] = np.zeros(num_classes)

    self.bn_params = [{'mode': 'train'} for i in range(2)]
    self.params['gamma1'] = np.ones(num_filters)
    self.params['beta1'] = np.zeros(num_filters)
    self.params['gamma2']  = np.ones(hidden_dim)
    self.params['beta2'] = np.zeros(hidden_dim)

    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):
    """
    Evaluate loss and gradient for the three-layer convolutional network.

    Input / output: Same API as TwoLayerNet in fc_net.py.
    """
    W1, b1 = self.params['W1'], self.params['b1']
    W2, b2 = self.params['W2'], self.params['b2']
    W3, b3 = self.params['W3'], self.params['b3']
    gamma1, beta1 = self.params['gamma1'], self.params['beta1']
    gamma2, beta2 = self.params['gamma2'], self.params['beta2']

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # pass
    conv, conv_cache = conv_norm_relu_pool_forward(X,W1,b1,conv_param, pool_param,
        gamma1, beta1, self.bn_params[0])
    h1, h1_cache = affine_norm_relu_forward(conv, W2, b2, gamma2, beta2,
        self.bn_params[1])
    scores, scores_cache = affine_forward(h1, W3, b3)
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # pass
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5* self.reg * (np.sum(W1*W1) + np.sum(W2*W2) + np.sum(W3*W3))
    loss = data_loss + reg_loss
    dh1, grads['W3'], grads['b3'] = affine_backward(dscores, scores_cache)
    dconv, grads['W2'], grads['b2'], grads['gamma2'], grads['beta2'] \
            = affine_norm_relu_backward(dh1, h1_cache)
    _, grads['W1'], grads['b1'], grads['gamma1'], grads['beta1'] \
        = conv_norm_relu_pool_backward(dconv, conv_cache)
    for k, v in grads.items():
        if k[0] == 'W':
            grads[k] += self.reg * v
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads



class convnet1(object):
  """
  A multi layered convolution network with architecture:
  [conv-relu-conv-relu-pool]xN - [affine]xM - [softmax or SVM]

  The network operates on minibatches of data that have shape (N, C, H, W)
  consisting of N images, each with height H and width W and with C input
  channels.
  """

  def __init__(self, hidden_dims, input_dim=(3, 32, 32), num_conv = 2,
            num_affine = 2,num_filters = 32, filter_size=7, num_classes=10,
            weight_scale=1e-3, reg=0.0, dtype= np.float32, use_batchnorm = True):
    """
    Initialize a new network.

    Inputs:
    - input_dim: Tuple (C, H, W) giving size of input data
    - num_filters: Number of filters to use in the convolutional layer
    - filter_size: Size of filters to use in the convolutional layer
    - hidden_dim: Number of units to use in the fully-connected hidden layer
    - num_classes: Number of scores to produce from the final affine layer.
    - weight_scale: Scalar giving standard deviation for random initialization
      of weights.
    - reg: Scalar giving L2 regularization strength
    - dtype: numpy datatype to use for computation.
    """
    self.params = {}
    self.reg = reg
    self.dtype = dtype
    self.bn_params = []
    self.num_conv = num_conv
    self.num_affine = num_affine
    self.use_batchnorm = use_batchnorm
    self.filter_size = filter_size

    ############################################################################
    # TODO: Initialize weights and biases for the three-layer convolutional    #
    # network. Weights should be initialized from a Gaussian with standard     #
    # deviation equal to weight_scale; biases should be initialized to zero.   #
    # All weights and biases should be stored in the dictionary self.params.   #.                                              #
    ############################################################################
    # pass
    C, H, W = input_dim
    num_channels, height, width = C, H , W
    for i in range(2*num_conv):

        char_i = str(i+1)
        W_i, b_i = 'W' + char_i, 'b' + char_i
        dim_i = (num_filters[i], num_channels, filter_size, filter_size)
        self.params[W_i] = weight_scale * np.random.rand(*dim_i)
        self.params[b_i] = np.zeros(num_filters[i])

        gamma_i, beta_i = 'gamma' + char_i , 'beta' + char_i
        self.params[gamma_i] = np.ones(num_filters[i])
        self.params[beta_i] = np.zeros(num_filters[i])
        num_channels = num_filters[i]

        if i % 2:
            height, width = height//2 , width//2

    dim =  height * width * num_channels
    for i in range(num_affine):
        char_i = str(i+2*num_conv+1)
        W_i , b_i = 'W' + char_i , b + char_i
        self.params[W_i] = weight_scale * np.random.randn(dim, hidden_dims[i])
        self.params[b_i] = np.zeros(hidden_dims[i])

        gamma_i, beta_i = 'gamma' + char_i , 'beta' + char_i
        self.params[gamma_i] = np.ones(hidden_dims[i])
        self.params[beta_i] = np.zeros(hidden_dims[i])

    if self.use_batchnorm:
        self.bn_params = [{'mode': 'train'} for i in range(2*num_conv + num_affine)]
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    for k, v in self.params.items():
      self.params[k] = v.astype(dtype)


  def loss(self, X, y=None):

    num_conv, num_affine = self.num_conv, self.num_affine

    # pass conv_param to the forward pass for the convolutional layer
    filter_size = W1.shape[2]
    conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

    # pass pool_param to the forward pass for the max-pooling layer
    pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

    scores = None
    ############################################################################
    # TODO: Implement the forward pass for the three-layer convolutional net,  #
    # computing the class scores for X and storing them in the scores          #
    # variable.                                                                #
    ############################################################################
    # pass
    conv_cache = [None for i in range(2*self.num_conv)]
    conv = X
    for i in range(2*num_conv):
        char_i = str(i+1)
        W_i, b_i , = 'W' + char_i, 'b' + char_i,
        gamma_i, beta_i = 'gamma' + char_i, 'beta' + char_i

        if i % 2:
            conv, conv_cache[i] = conv_norm_relu_pool_forward(conv, self.params[W_i],
             self.params[b_i], conv_param, pool_param, gamma_i, beta_i, self.bn_params[i])
        else:
            conv, conv_cache[i] = conv_norm_relu_forward(conv, self.params[W_i],
                self.params[b_i], conv_param, gamma_i, beta_i, self.bn_params[i])

    affine_cache = [None for i in range(num_affine)]
    out = conv
    for i in range(num_affine):
        char_i = str(i + 2*num_conv + 1)
        W_i, b_i = 'W' + char_i, 'b' + char_i,
        gamma_i, beta_i = 'gamma' + char_i, 'beta' + char_i
        out, affine_cache[i] = affine_norm_forward(out, self.params[W_i],
                self.params[b_i], gamma_i, beta_i, self.bn_params[i])

    scores = out
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    if y is None:
      return scores

    loss, grads = 0, {}
    ############################################################################
    # TODO: Implement the backward pass for the three-layer convolutional net, #
    # storing the loss and gradients in the loss and grads variables. Compute  #
    # data loss using softmax, and make sure that grads[k] holds the gradients #
    # for self.params[k]. Don't forget to add L2 regularization!               #
    ############################################################################
    # pass
    data_loss, dscores = softmax_loss(scores, y)
    reg_loss = 0.5 * self.reg * np.sum([np.sum(self.params['W' + str(i+1)]**2) \
                for i in range(num_conv*2 + num_affine)])
    loss = data_loss + reg_loss
    dh = dscores
    for i in range(num_affine, 0, -1):
        char_i = str(i + 2* num_conv)
        W_i, b_i , = 'W' + char_i, 'b' + char_i,
        gamma_i, beta_i = 'gamma' + char_i, 'beta' + char_i
        dh, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = \
        affine_norm_backward(dh, affine_cache[i-1])

    dconv = dh
    for i in range(2*num_conv, 0, -1):
        char_i = str(i)
        W_i, b_i , = 'W' + char_i, 'b' + char_i,
        gamma_i, beta_i = 'gamma' + char_i, 'beta' + char_i
        if i % 2 == 0:
            dconv, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = \
            conv_norm_relu_pool_backward(dconv, conv_cache[i-1])
        else:
            dconv, grads[W_i], grads[b_i], grads[gamma_i], grads[beta_i] = \
            conv_norm_relu_backward(dconv, conv_cache[i-1])

    for k, v in grads.items():
        if k[0] == 'W':
            grads[k] += self.reg * v
    ############################################################################
    #                             END OF YOUR CODE                             #
    ############################################################################

    return loss, grads


# pass