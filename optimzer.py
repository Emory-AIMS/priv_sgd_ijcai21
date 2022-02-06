import tensorflow as tf
import dp_optimizer


def SGD(loss_op, lr, momentum):
    optimizer = tf.train.MomentumOptimizer(learning_rate=lr, momentum=momentum)
    update_ops_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_bn):
        train_op = optimizer.minimize(loss_op)
    return train_op


def DPSGD(loss_op, lr, momentum, l2_norm_clip, noise_multiplier, microbatches):
    optimizer = dp_optimizer.DPMomentumGaussianOptimizer(
        l2_norm_clip=l2_norm_clip,
        noise_multiplier=noise_multiplier,
        num_microbatches=microbatches,
        learning_rate=lr,
        momentum=momentum)
    update_ops_bn = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops_bn):
        train_op = optimizer.minimize(loss_op)
    return train_op