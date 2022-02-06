# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
import tensorflow as tf
import numpy as np
from datetime import datetime
from resnet_model import resnet_inference
import os
import cifar_input as cifar_data
import my_utils
import optimzer as opt

tf.logging.set_verbosity(tf.logging.WARN)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.reset_default_graph()

try:
    FLAGS.activation
except:
    FLAGS = tf.app.flags.FLAGS
    tf.app.flags.DEFINE_string('activation', 'elu', '''activation function to use: relu or elu''')
    tf.app.flags.DEFINE_integer('random_seed', 123, '''random seed for initialization''')
    tf.app.flags.DEFINE_integer('train_batch_size', 32, '''batch_size''')
    tf.app.flags.DEFINE_float('lr', 0.1, '''learning rate to train the models''')
    tf.app.flags.DEFINE_integer('dataset', 10, '''dataset to evalute: 10 or 100''')
    tf.app.flags.DEFINE_integer('resnet_layers', 20,
                                '''number of layers to use in ResNet: 56 or 20''')
    tf.app.flags.DEFINE_integer('num_iters', 100 * 400, '''total number of iterations to train the model''')
    tf.app.flags.DEFINE_boolean('is_tune', False,
                                '''if True, split train dataset (50K) into 45K, 5K as train/validation data''')
    tf.app.flags.DEFINE_boolean('is_save_model', False, '''whether to save model or not ''')
    tf.app.flags.DEFINE_boolean('use_L2', False, '''whether to use L2 regularizer''')
    tf.app.flags.DEFINE_integer('version', 2,  '''1: non-dp SGD, stagewise lr;
                                                  2: dp SGD, constant lr; 
                                                  3: dp SGD, c/t;
                                                  4: dp SGD, c/sqrt(t);
                                                  5: dp SGD momentum, constant lr;
                                                  6: dp SGD early momentum, constant lr;
                                                  7: dp SGD, stagewise lr;
                8: dp SGD early momentum, stagewise lr: each stage turns on/off momentum''')

    tf.app.flags.DEFINE_float('l2_norm_clip', 3.0, 'Clipping norm')
    tf.app.flags.DEFINE_float('momentum', 0.9, 'Momentum coefficient for MomentumOptimizer')
    tf.app.flags.DEFINE_float('epsilon', 10, 'Privacy budget epsilon')
    tf.app.flags.DEFINE_float('early_momentum_ratio', 0.5, 'Ratio of iterations in each stage for early momentum (for version 8)')
    tf.app.flags.DEFINE_integer('early_momentum_iters', 100*200, 'Number of iterations in early momentum stage (for version 6)')
    tf.app.flags.DEFINE_integer('decay_factor', 2, 'learning rate decay factor for each stage (for version 1, 7, 8, 9)')
    tf.app.flags.DEFINE_integer('microbatches', 32, 'Number of microbatches (must evenly divide train_batch_size)')


tf.set_random_seed(FLAGS.random_seed)


def calculate_noise_multiplier():
    delta = float(1.0 / 50000)
    num_iter = FLAGS.num_iters
    sigma = FLAGS.train_batch_size * np.sqrt((8 * num_iter * (FLAGS.l2_norm_clip ** 2) * np.log(1 / delta)) / (50000 ** 2 * FLAGS.epsilon ** 2))
    noise_multiplier = sigma / FLAGS.l2_norm_clip
    return noise_multiplier


def eval_once(images, labels):
    num_batches = images.shape[0] // batch_size
    accuracy_mean = []
    loss_mean = []
    for step in range(num_batches):
        offset = step * batch_size
        vali_data_batch = images[offset:offset + batch_size]
        vali_label_batch = labels[offset:offset + batch_size]
        loss, acc = sess.run([loss_op, accuracy],
                             feed_dict={X: vali_data_batch, Y: vali_label_batch, phase_train: False})
        accuracy_mean.append(acc)
        loss_mean.append(loss)
    return np.mean(loss_mean), np.mean(accuracy_mean)


# Import CIFAR data
(train_data, train_labels), (test_data, test_labels) = cifar_data.load_data(FLAGS.dataset, FLAGS.is_tune)

# Training Parameters
initial_learning_rate = FLAGS.lr
num_iters = FLAGS.num_iters
batch_size = FLAGS.train_batch_size
inference = resnet_inference

# Network Parameters
if FLAGS.version in [2, 3, 4]:
    result_file_name = "cifar10_method-{}_lr-{}_epsilon-{}_clip-{}_batch-{}_iterations-{}.txt".format(FLAGS.version,
                                                                                               FLAGS.lr,
                                                                                               FLAGS.epsilon,
                                                                                               FLAGS.l2_norm_clip,
                                                                                               FLAGS.train_batch_size,
                                                                                               FLAGS.num_iters)
elif FLAGS.version in [5, 6]:
    result_file_name = "cifar10_method-{}_lr-{}_momentum-{}_earlyMIters-{}_epsilon-{}_clip-{}_batch-{}_iterations-{}.txt". \
        format(FLAGS.version, FLAGS.lr, FLAGS.momentum, FLAGS.early_momentum_iters,
               FLAGS.epsilon, FLAGS.l2_norm_clip, FLAGS.train_batch_size, FLAGS.num_iters)
elif FLAGS.version == 1:
    result_file_name = "cifar10_method-{}_lr-{}_batch-{}_iterations-{}_decayFactor-{}.txt".format(FLAGS.version,
                                                                                               FLAGS.lr,
                                                                                               FLAGS.train_batch_size,
                                                                                               FLAGS.num_iters,
                                                                                               FLAGS.decay_factor)
elif FLAGS.version == 7:
    result_file_name = "cifar10_method-{}_lr-{}_epsilon-{}_clip-{}_batch-{}_iterations-{}_decayFactor-{}.txt". \
            format(FLAGS.version, FLAGS.lr, FLAGS.epsilon, FLAGS.l2_norm_clip,
                   FLAGS.train_batch_size, FLAGS.num_iters, FLAGS.decay_factor)
elif FLAGS.version == 8:
    result_file_name = "cifar10_method-{}_lr-{}_momentum-{}_earlyMRatio-{}_epsilon-{}_clip-{}_batch-{}_iterations-{}_decayFactor-{}.txt". \
            format(FLAGS.version, FLAGS.lr, FLAGS.momentum, FLAGS.early_momentum_ratio,
                   FLAGS.epsilon, FLAGS.l2_norm_clip, FLAGS.train_batch_size, FLAGS.num_iters, FLAGS.decay_factor)
else:
    exit("None Defined Version.\n")


noise_multiplier = calculate_noise_multiplier()
print("noise multiplier = {} when epsilon = {}".format(noise_multiplier, FLAGS.epsilon))

# create tf Graph input
X = tf.placeholder(tf.float32, [batch_size, 32, 32, 3])
Y = tf.placeholder(tf.float32, [batch_size, ])
lr = tf.placeholder(tf.float32)
momentum = tf.placeholder(tf.float32)
phase_train = tf.placeholder(tf.bool, name='phase_train')

# Construct model
logits = inference(X, num_classes=FLAGS.dataset, num_layers=FLAGS.resnet_layers, activations=FLAGS.activation,
                   phase_train=phase_train)  # when resnet you need to pass number of layers
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
W = [var for var in tf.trainable_variables()]
if FLAGS.version in [2, 3, 4, 5, 6, 7, 8]:
    loss_op = my_utils.cross_entropy_loss_with_l2_vectorized(logits, Y)
    train_op = opt.DPSGD(loss_op, lr=lr, momentum=momentum, l2_norm_clip=FLAGS.l2_norm_clip,
                         noise_multiplier=noise_multiplier, microbatches=FLAGS.microbatches)
else:
    loss_op = my_utils.cross_entropy_loss_with_l2(logits, Y, W, use_L2=FLAGS.use_L2)
    train_op = opt.SGD(loss_op, lr=lr, momentum=momentum)

lr_s, momentum_s, _, _ = my_utils.learning_rate_decay(FLAGS.version, initial_learning_rate, FLAGS.momentum, num_iters,
                                        FLAGS.decay_factor, FLAGS.early_momentum_iters, FLAGS.early_momentum_ratio)

# Evaluate model
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# store models
saver = tf.train.Saver(tf.global_variables(), max_to_keep=5000)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    # Run the initializer
    sess.run(init)
    print('Check W0: [%.3f]' % (sess.run(W))[0].sum())
    print('\nStart training...')

    for iter_ in range(num_iters):

        learning_rate = lr_s[iter_]
        curr_momentum = momentum_s[iter_]
        batch_x, batch_y = cifar_data.generate_augment_train_batch(train_data, train_labels, batch_size, FLAGS.is_tune)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, lr: learning_rate, momentum: curr_momentum, phase_train: True})

        if iter_ % 400 == 0:
            # Calculate loss and accuracy over entire dataset
            train_loss, train_acc = eval_once(train_data, train_labels)
            test_loss, test_acc = eval_once(test_data, test_labels)
            print("%s: [%d]: lr:%.5f, Train_Loss: %.5f, Test_loss: %.5f, Train_acc:, %.5f, Test_acc:, %.5f" % (
            datetime.now(), iter_, learning_rate, train_loss, test_loss, train_acc, test_acc))

            generalization_error = np.abs(train_acc - test_acc)
            result_file = open(result_file_name, "a")  # append mode
            result_file.write(
                "{}, {}, {}, {}, {}, {}\n".format(iter_, train_acc, test_acc, generalization_error, learning_rate, curr_momentum))
            result_file.close()

            if not FLAGS.is_tune and FLAGS.is_save_model:
                if FLAGS.version == 1:
                    save_dir = './models-%d_v%d_%s_L2_%s/C%d/lr_%d_batch_%d/' % (
                        FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2),
                        FLAGS.dataset, FLAGS.lr, FLAGS.train_batch_size)
                else:
                    save_dir = './models-%d_v%d_%s_L2_%s/C%d/lr_%d_eps_%d_clip_%d_batch_%d_decay_%d/' % (
                        FLAGS.resnet_layers, FLAGS.version, FLAGS.activation, str(FLAGS.use_L2),
                        FLAGS.dataset, FLAGS.lr, FLAGS.epsilon, FLAGS.l2_norm_clip, FLAGS.train_batch_size, FLAGS.decay_factor)
                checkpoint_path = os.path.join(save_dir, 'model-%d_v%d_cifar%d_%s.ckpt' % (
                FLAGS.resnet_layers, FLAGS.version, FLAGS.dataset, FLAGS.activation))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                if iter_ < FLAGS.num_iters or iter_ > 199600:
                    saver.save(sess, checkpoint_path, global_step=iter_, write_meta_graph=False)
