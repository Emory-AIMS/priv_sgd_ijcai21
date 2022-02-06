import tensorflow as tf
import numpy as np
import re


def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)


def cross_entropy_loss_with_l2(logits, labels, W=[], weight_decay=0.0005, use_L2=True):
    labels = tf.cast(labels, tf.int64)
    loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    print ('use_L2: [{}]'.format(use_L2))
    if use_L2:
        var_list_no_bias = [var for var in W if len(var.get_shape().as_list()) != 1] # no bias added
        l2_loss = tf.add_n([tf.nn.l2_loss(var) for idx, var in enumerate(var_list_no_bias)])
        loss_op = loss_op + l2_loss*weight_decay
    return loss_op


def cross_entropy_loss_with_l2_vectorized(logits, labels):
    labels = tf.cast(labels, tf.int64)
    loss_op = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    return loss_op


def create_placeholders(params):
    weights_holder = []
    for idx, item in enumerate(params):
        tensor_shape = item.get_shape().as_list()
        w = tf.placeholder(tf.float32, tensor_shape, name='W_avg_%d'%idx)
        weights_holder.append(w)
    return weights_holder

def compute_average_weights(W_mean, W_t, t):
    new_W_mean = []
    for idx, w_bar in enumerate(W_mean):
        w_bar_new =  (w_bar*t + W_t[idx])/(t + 1)  # make sure t0 = 0, otherwise, we need to chaneg t to t-1
        new_W_mean.append(w_bar_new)
    return new_W_mean


def cumulative_sum(W_mean, W_t):
        new_W_mean = []
        for idx, w_bar in enumerate(W_mean):
            new_w_bar = w_bar + W_t[idx]
            new_W_mean.append(new_w_bar)
        return new_W_mean
    

def learning_rate_decay(version, learning_rate, momentum, num_iters, decay_factor, early_momentum_iters,
                        early_momentum_ratio):
    print ('Learning_decay: [v%d]'%version)
    T_s = [t for t in range(num_iters)]
    num_S = len(T_s)
    if version == 1:  # non-dp SGD, stagewise lr
        lr_s = [learning_rate]*20000 + [learning_rate/decay_factor]*10000 + [learning_rate/decay_factor**2]*100000
        momentum_s = [0 for _ in range(1, num_iters+1)]
    if version == 2:  # dp SGD, constant lr
        lr_s = [learning_rate for _ in range(1, num_iters+1)]
        momentum_s = [0 for _ in range(1, num_iters+1)]
    if version == 3:  # dp SGD, c/t
        lr_s = [learning_rate/t for t in range(1, num_iters+1)]
        momentum_s = [0 for _ in range(1, num_iters+1)]
    if version == 4:  # dp SGD, c/sqrt(t)
        lr_s = [learning_rate/np.sqrt(t) for t in range(1, num_iters+1)]
        momentum_s = [0 for _ in range(1, num_iters+1)]
    if version == 5:  # dp SGD momentum, constant lr
        lr_s = [learning_rate for _ in range(1, num_iters+1)]
        momentum_s = [momentum for _ in range(1, num_iters+1)]
    if version == 6:  # dp SGD early momentum, constant lr
        lr_s = [learning_rate for _ in range(1, num_iters+1)]
        momentum_s = [momentum]*early_momentum_iters + [0]*100000
    if version == 7:  # dp SGD, stagewise lr
        lr_s = [learning_rate]*20000 + [learning_rate/decay_factor]*10000 + [learning_rate/decay_factor**2]*100000
        momentum_s = [0 for _ in range(1, num_iters+1)]
    if version == 8:  # dp SGD early momentum, stagewise lr (each stage turns on/off momentum)
        lr_s = [learning_rate]*20000 + [learning_rate/decay_factor]*10000 + [learning_rate/decay_factor**2]*100000
        momentum_s = [momentum]*(int(20000*early_momentum_ratio)) + [0]*(20000-int(20000*early_momentum_ratio)) \
                     + [momentum]*(int(10000*early_momentum_ratio)) + [0]*(10000-int(10000*early_momentum_ratio)) \
                     + [momentum]*(int(10000*early_momentum_ratio)) + [0]*100000
    return lr_s, momentum_s, T_s, num_S