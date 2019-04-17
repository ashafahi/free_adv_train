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
from free_model import Model
from pgd_attack import LinfPGDAttack, GaussianNoiseAugmentation, UniformNoiseAugmentation
import cifar10_input


with open('config.json') as config_file:
    config = json.load(config_file)


assert (config['size_uniform'] == 0.0) or (config['std_rand'] == 0.0), 'you are both doing uniform and gaussian augmentation!'
_RESET_PERTURB = config['reset_perturbation']
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


# Setting up the optimizer
boundaries = [int(sss[0]) for sss in step_size_schedule]
boundaries = boundaries[1:]
values = [sss[1] for sss in step_size_schedule]
learning_rate = tf.train.piecewise_constant(
    tf.cast(global_step, tf.int32),
    boundaries,
    values)

total_loss = model.mean_xent + weight_decay * model.weight_decay_loss
min_optim = tf.train.MomentumOptimizer(learning_rate, momentum)
gradzzz = min_optim.compute_gradients(total_loss)
pert_grad = [g for g,v in gradzzz if 'perturbation' in v.name]
no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g,v) for g,v in gradzzz]
sign_pert_grad = tf.sign(pert_grad[0])
new_pert = model.pert + config['epsilon']*sign_pert_grad
clip_new_pert = tf.clip_by_value(new_pert, -config['epsilon'], config['epsilon'])
assignet = tf.assign(model.pert, clip_new_pert)
with tf.control_dependencies([assignet]):
  min_step = min_optim.apply_gradients(no_pert_grad, global_step=global_step)
reset_perturb = tf.initialize_variables([model.pert])

rand_aug = GaussianNoiseAugmentation(mean=0.0, std=std_gauss, random_start=True, do_clipping=True)

uniform_aug = UniformNoiseAugmentation(size=config['size_uniform'], random_start=True, do_clipping=True)

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
  l2_dists, linf_dists, sign_pert_grads = list(), list(), list()
  prev_spg = np.zeros((batch_size,32*32*3)).reshape(batch_size,-1)
  for ii in range(max_num_training_steps):
    if ii % config['num_steps_skip'] == 0:
      if _RESET_PERTURB == True:
        sess.run(reset_perturb)
      x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)

      # Compute Adversarial Perturbations on random gaussian augmented images
      start = timer()
      x_batch_adv = rand_aug.perturb(x_batch)
      x_batch_adv = uniform_aug.perturb(x_batch_adv)
      end = timer()
    diff = (sess.run(model.final_input, feed_dict={model.x_input:x_batch_adv}) - x_batch).reshape(batch_size,-1)
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
    #sess.run(train_step, feed_dict=adv_dict)
    #sess.run(min_step, feed_dict=adv_dict)
    spg, _ = sess.run([sign_pert_grad, min_step], feed_dict=adv_dict)
    spg = spg.reshape(batch_size, -1)
    diff_spg = np.sum((prev_spg==spg).astype(np.int8), axis=1)
    sign_pert_grads.append(diff_spg)
    prev_spg = spg
    end = timer()
    training_time += end - start
  np.save(model_dir.split('/')[-1]+'_l2.npy',np.array(l2_dists))
  np.save(model_dir.split('/')[-1]+'_linf.npy',np.array(linf_dists))
  np.save(model_dir.split('/')[-1]+'_spg.npy',np.array(sign_pert_grads))
