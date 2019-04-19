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
import sys
from free_model import Model
import cifar10_input
import pdb

import config

args = config.get_args()


def get_path_dir(data_dir, dataset, **_):
    path = os.path.join(data_dir, dataset)
    if os.path.islink(path):
        path = os.readlink(path)
    return path


def train(tf_seed, np_seed, train_steps, out_steps, summary_steps, checkpoint_steps, step_size_schedule,
          weight_decay, momentum, train_batch_size, epsilon, replay_m, model_dir, **kwargs):
    # Setting up training parameters
    tf.set_random_seed(tf_seed)
    np.random.seed(np_seed)
    # Setting up training parameters
    max_num_training_steps = train_steps
    num_output_steps = out_steps
    num_summary_steps = summary_steps
    num_checkpoint_steps = checkpoint_steps
    step_size_schedule = step_size_schedule
    weight_decay = weight_decay

    data_path = get_path_dir(**kwargs)
    print(data_path)
    momentum = momentum
    batch_size = train_batch_size
    epsilon = epsilon
    replay_m = replay_m
    model_dir = model_dir + '_m%d_eps%.1f_b%d' % (replay_m, epsilon, batch_size)

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

    # optimizing computation
    min_optim = tf.train.MomentumOptimizer(learning_rate, momentum)
    gradientz = min_optim.compute_gradients(total_loss)
    pert_grad = [g for g, v in gradientz if 'perturbation' in v.name]
    no_pert_grad = [(tf.zeros_like(v), v) if 'perturbation' in v.name else (g, v) for g, v in gradientz]
    sign_pert_grad = tf.sign(pert_grad[0])
    new_pert = model.pert + epsilon * sign_pert_grad
    clip_new_pert = tf.clip_by_value(new_pert, -epsilon, epsilon)
    assignet = tf.assign(model.pert, clip_new_pert)
    with tf.control_dependencies([assignet]):
        min_step = min_optim.apply_gradients(no_pert_grad, global_step=global_step)
    reset_perturb = tf.initialize_variables([model.pert])

    # Setting up the Tensorboard and checkpoint outputs
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    saver = tf.train.Saver(max_to_keep=1)
    tf.summary.scalar('accuracy', model.accuracy)
    tf.summary.scalar('xent', model.xent / batch_size)
    tf.summary.scalar('total loss', total_loss / batch_size)
    merged_summaries = tf.summary.merge_all()

    shutil.copy('config.json', model_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1.0)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        print(
            '\n\n************ free training for epsilon=%.1f using m_replay=%d *************\n\n' % (epsilon, replay_m))
        # initialize data augmentation
        cifar = cifar10_input.AugmentedCIFAR10Data(raw_cifar, sess, model)

        # Initialize the summary writer, global variables, and our time counter.
        summary_writer = tf.summary.FileWriter(model_dir + '/train', sess.graph)
        eval_summary_writer = tf.summary.FileWriter(model_dir + '/eval')
        sess.run(tf.global_variables_initializer())

        # Main training loop
        for ii in range(max_num_training_steps):
            if ii % replay_m == 0:
                x_batch, y_batch = cifar.train_data.get_next_batch(batch_size, multiple_passes=True)

                nat_dict = {model.x_input: x_batch,
                            model.y_input: y_batch}

            x_eval_batch, y_eval_batch = cifar.eval_data.get_next_batch(batch_size,
                                                                        multiple_passes=True)

            eval_dict = {model.x_input: x_eval_batch,
                         model.y_input: y_eval_batch}
            # Output to stdout
            if ii % num_summary_steps == 0:
                train_acc, summary = sess.run([model.accuracy, merged_summaries], feed_dict=nat_dict)
                summary_writer.add_summary(summary, global_step.eval(sess))
                val_acc, summary = sess.run([model.accuracy, merged_summaries], feed_dict=eval_dict)
                eval_summary_writer.add_summary(summary, global_step.eval(sess))
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}% -- validation nat accuracy {:.4}%'.format(train_acc * 100,
                                                                                                  val_acc * 100))
                sys.stdout.flush()
            # Tensorboard summaries
            elif ii % num_output_steps == 0:
                nat_acc = sess.run(model.accuracy, feed_dict=nat_dict)
                print('Step {}:    ({})'.format(ii, datetime.now()))
                print('    training nat accuracy {:.4}%'.format(nat_acc * 100))

            # Write a checkpoint
            if ii % num_checkpoint_steps == 0:
                saver.save(sess,
                           os.path.join(model_dir, 'checkpoint'),
                           global_step=global_step)

            # Actual training step
            sess.run(min_step, feed_dict=nat_dict)


if __name__ == '__main__':
    train(**vars(args))
