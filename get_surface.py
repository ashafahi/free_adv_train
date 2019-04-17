"""Evaluates a model against examples from a .npy file as specified
   in config.json"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import math

import numpy as np
import tensorflow as tf

import cifar10_input
from model import Model

madry_pretrained_models = ['natural', 'adv_trained']

with open('config_plotting.json') as config_file:
    config = json.load(config_file)

data_path = config['data_path']
cifar = cifar10_input.CIFAR10Data(data_path)
_BATCH_SIZE = config['batch_size']

# seeding randomness
tf.set_random_seed(config['tf_random_seed'])
np.random.seed(config['np_random_seed'])
epsilon = config['epsilon']
buffer_eps = config['eps_buffer']
step_interval = config['plot_step_size']
dir_2_rand = config['dir_2_rand']

frd = (np.random.rand(32,32,3)>=0.5).astype(np.float32)
frd[frd==0.0]=-1


model_dir = config['model_dir']
saving_name = 'surface_numpy_files/' + model_dir.split('/')[-1]+'_pl'+str(buffer_eps+epsilon)+'_'+str(step_interval)+(not dir_2_rand)*'adv'+'.npy' # name of file for saving the outplot of this script
if model_dir.split('/')[-1] in madry_pretrained_models:
  model = Model(mode='eval')
  print('loading the intact model from madrylabs github repo')
else:
  model = Model(mode='eval')
saver = tf.train.Saver()
checkpoint = tf.train.latest_checkpoint(model_dir)
sess =  tf.Session()
saver.restore(sess, checkpoint)

gridz = np.arange(-epsilon-buffer_eps, epsilon+buffer_eps+step_interval, step_interval)
dir_1_rand = True
if dir_2_rand:
  frd2 = (np.random.rand(32,32,3)>=0.5).astype(np.float32)
  frd2[frd2==0.0]=-1

def get_for_example(example_id, model=model, sess=sess, cifar=cifar):
    gradient_dir = tf.gradients(model.y_xent, model.x_input)[0]

    # example configurations
    x_nat = cifar.eval_data.xs[example_id]
    y_nat = cifar.eval_data.ys[example_id] 

    if dir_2_rand:
       grd = frd2
    else:
       grd = np.squeeze(np.sign(sess.run(gradient_dir, feed_dict={model.x_input: np.expand_dims(x_nat, axis=0), model.y_input: np.expand_dims(y_nat, axis=0)})))

    # now do a grid search run on the stepsize parameters
    x1_deltas = gridz    
    x2_deltas = gridz    
    
    total_num_points = len(x1_deltas)*len(x2_deltas)
    num_points = 0
    x_batch = []
    y_batch = []
    x1s = []
    x2s = []
    all_losses = []
    for i, x1 in enumerate(x1_deltas):
      for j, x2 in enumerate(x2_deltas):
        num_points = i*len(x2_deltas) + j + 1
        x_batch.append(x_nat + x1*frd + x2*grd)
        y_batch.append(y_nat)
        x1s.append(x1)
        x2s.append(x2)
        if (num_points % _BATCH_SIZE == 0) or (num_points >= total_num_points):
          x_batch = np.array(x_batch)
          y_batch = np.array(y_batch) 
          losses = sess.run(model.y_xent, feed_dict={model.x_input: x_batch, model.y_input: y_batch})
          x_batch = []
          y_batch = []
          all_losses.append(losses)
          
    all_losses = np.squeeze(np.concatenate(np.ravel(all_losses)))
        
    return [(i1,i2,i3) for i1,i2,i3 in zip(x1s,x2s,all_losses)]
       

if __name__ == '__main__':


    if checkpoint is None:
        print('No checkpoint found')
    else:
        all_rez = []
        for ex in range(8):
           res = get_for_example(example_id=ex)
           all_rez.append(res)
           print('finished example %d'%ex)
           all_rez_np = np.array(all_rez)
           np.save(saving_name, all_rez)
