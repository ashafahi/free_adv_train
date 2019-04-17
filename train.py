"""Trains a model, saving checkpoints and tensorboard summaries along
   the way."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import json
import os
import shutil
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import sys
from model import Model
from pgd_attack import LinfPGDAttack, GaussianNoiseAugmentation, UniformNoiseAugmentation
import cifar10_input

_REGULARIZER_ON = 'logits'
_REGULARIZER_TYPE = 'l2'
_RANDOM_TRAIN = True#False
_COEF_FINALWEIGHTS = 0.
_THRESH_WEIGHTS = 0.#2.
_THRESH_REG_TERM = 0.#20.

assert _REGULARIZER_ON in ['logits','features', None], 'the regularizer selected: %s does not belong to the acceptable list'%str(_REGULARIZER_ON)
assert _REGULARIZER_TYPE in ['l1', 'linf', 'l2'], 'the regularizer type: %s does not belong to the acceptable list'%str(_REGULARIZER_TYPE)
if _REGULARIZER_ON is None:
    _REGULARIZER_COEF = 0.


with open('config.json') as config_file:
    config = json.load(config_file)


assert (config['size_uniform'] == 0.0) or (config['std_rand'] == 0.0), 'you are both doing uniform and gaussian augmentation!'

# Setting up training parameters
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])
# Setting up training parameters
max_num_training_steps = config['max_num_training_steps']
num_output_steps = config['num_output_steps']
num_summary_steps = config['num_summary_steps']
num_checkpoint_steps = config['num_checkpoint_steps']
step_size_schedule = config['step_size_schedule']
weight_decay = config['weight_decay']
data_path = config['data_path']
momentum = config['momentum']
batch_size = config['training_batch_size']
std_gauss = config['std_rand']

# Setting up the data and the model
raw_cifar = cifar10_input.CIFAR10Data(data_path)
global_step = tf.contrib.framework.get_or_create_global_step()
model = Model(mode='train')

# adding the regularization
regularizer_map = {'logits': model.pre_softmax, 'features': model.neck, None: 0.}
ord_map = {'l1':1, 'l2': 2, 'linf':np.inf}

regularized_term = regularizer_map[_REGULARIZER_ON]
ord_term = ord_map[_REGULARIZER_TYPE]
regularizer = tf.reduce_sum(tf.nn.relu(tf.norm(regularized_term, ord=ord_term) - _THRESH_REG_TERM))# , axis=1

# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)
total_loss = model.mean_xent + weight_decay * model.weight_decay_loss 
train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(
    total_loss,
    global_step=global_step)


# Set up adversary
attack = LinfPGDAttack(model, 
                       config['epsilon'],
                       config['num_steps'],
                       config['step_size'],
                       config['random_start'],
                       config['loss_func'])


rand_aug = GaussianNoiseAugmentation(mean=0.0, std=std_gauss, random_start=_RANDOM_TRAIN, do_clipping=True)

uniform_aug = UniformNoiseAugmentation(size=config['size_uniform'], random_start=_RANDOM_TRAIN, do_clipping=True)

# Setting up the Tensorboard and checkpoint outputs
model_dir = config['model_dir']
if not os.path.exists(model_dir):
  os.makedirs(model_dir)

# We add accuracy and xent twice so we can easily make three types of
# comparisons in Tensorboard:
# - train vs eval (for a single run)
# - train of different runs
# - eval of different runs

saver = tf.train.Saver(max_to_keep=1)
tf.summary.scalar('accuracy adv train', model.accuracy)
tf.summary.scalar('accuracy adv', model.accuracy)
tf.summary.scalar('xent adv train', model.xent / batch_size)
tf.summary.scalar('xent adv', model.xent / batch_size)
tf.summary.scalar('total loss adv', total_loss / batch_size)
tf.summary.image('images adv train', model.x_input)
merged_summaries = tf.summary.merge_all()

shutil.copy('config.json', model_dir)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
  # initialize data augmentation
  cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)


  # Initialize the summary writer, global variables, and our time counter.
  summary_writer = tf.summary.FileWriter(model_dir, sess.graph)
  sess.run(tf.global_variables_initializer())
  training_time = 0.0

  # Main training loop
  l2_dists, linf_dists = list(), list()
  for ii in range(max_num_training_steps):
    if ii % config['num_steps_skip'] == 0:
      x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)

      # Compute Adversarial Perturbations on random gaussian augmented images
      start = timer()
      x_batch_adv = rand_aug.perturb(x_batch)
      x_batch_adv = uniform_aug.perturb(x_batch_adv)
      x_batch_adv = attack.perturb(x_batch_adv, y_batch, sess)
      end = timer()
      diff = (x_batch_adv - x_batch).reshape(batch_size,-1)
      l2_dists.append(np.linalg.norm(diff,axis=1, ord=2))
      linf_dists.append(np.linalg.norm(diff,axis=1, ord=np.inf))
      training_time += end - start

      nat_dict = {model.x_input: x_batch,
                model.y_input: y_batch}#,
 #               model.training: False}

      adv_dict = {model.x_input: x_batch_adv,
                model.y_input: y_batch}#,
             #   model.training: False}

    # Output to stdout
    if ii % num_output_steps == 0:
      nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
      adv_acc = sess.run(model.accuracy, feed_dict=adv_dict)
      print('Step {}:    ({})'.format(ii, datetime.now()))
      print('    training nat accuracy {:.4}%'.format(nat_acc * 100))
      print('    training adv accuracy {:.4}%'.format(adv_acc * 100))
      if ii != 0:
        print('    {} examples per second'.format(
            num_output_steps * batch_size / training_time))
        training_time = 0.0
      sys.stdout.flush()
    # Tensorboard summaries
    if ii % num_summary_steps == 0:
      summary = sess.run(merged_summaries, feed_dict=adv_dict)
      summary_writer.add_summary(summary, global_step.eval(sess))

    # Write a checkpoint
    if ii % num_checkpoint_steps == 0:
      saver.save(sess,
                 os.path.join(model_dir, 'checkpoint'),
                 global_step=global_step)

    # Actual training step
    start = timer()
#    adv_dict[model.training] = True
    sess.run(train_step, feed_dict=adv_dict)
    end = timer()
    training_time += end - start
  np.save(model_dir.split('/')[-1]+'_l2.npy',np.array(l2_dists))
  np.save(model_dir.split('/')[-1]+'_linf.npy',np.array(linf_dists))
