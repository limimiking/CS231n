import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  for i in xrange(num_train):
    f = X[i].dot(W)
    f -= np.max(f)
    scores = np.exp(f)
    scores = scores / np.sum(scores)
    correct_score = scores[y[i]]
    loss += -np.log(correct_score / np.sum(scores))
#     dW += X[i][:,None].dot( (scores - np.array(range(num_classes) == y[i]))[None,:])
    for j in xrange(num_classes):
      dW[:, j] += (scores[j] - (j == y[i]) * 1.0) * X[i]
  loss /= num_train
  dW /= num_train
  loss += 0.5 * reg * np.sum(W * W)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  num_train = X.shape[0]
  num_classes = W.shape[1]
  y_onehot = np.array(range(num_classes)) == y.reshape(-1,1)
  f = X.dot(W)
  f -= np.max(f, axis=1, keepdims=True)
  scores = np.exp(f)
  scores = scores / np.sum(scores, axis=1, keepdims=True)
  correct_score = scores[np.arange(num_train),y] # np.sum(scores * y_onehot, axis=1)
  loss = 1 / num_train * np.sum(-np.log(correct_score / np.sum(scores, axis=1)))
  loss += 0.5 * reg * np.sum(W * W)
  dW = 1 / num_train * X.T.dot(scores - y_onehot)
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

