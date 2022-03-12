from collections import deque

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

import config
import model


class A3CAgent:
   def __init__(self, memory_size, input_size, output_size, discount):
       self.input_size = input_size 
       self.output_size = output_size
       self.memory_size = memory_size
       self.discount = discount
       self.memory = deque()
       self.policy_model = model.get_policy_model(self.input_size, self.output_size)
       self.value_model = model.get_value_model(self.input_size)
   
   def remember(self, trajectory):
       if len(self.memory)>config.MEM_SIZE:
           self.memory.popleft()
       self.memory.append(trajectory)
       
   def forget(self):
       self.memory.clear()
       
   @tf.function
   def act(self, state):
       with tf.GradientTape() as tape:
           action_params = self.policy_model(state)
           batch_size = action_params.shape[0]
           mean = action_params[:,:self.output_size]
           cov_diag = tf.clip_by_value(action_params[:, self.output_size:],-4, 1)
           
           cov_diag =  tf.exp(cov_diag)
           dis = tfp.distributions.MultivariateNormalDiag(loc=mean,
                                                          scale_diag=cov_diag,
                                                          validate_args=True,
                                                          allow_nan_stats=False)
           action = tf.nn.tanh(dis.sample())
           pdf = dis.prob(action)
           log_pdf = tf.math.log(pdf)
       grads = tape.gradient(log_pdf, self.policy_model.trainable_weights)
       
       return action, grads
   
   def train_value_model(self, learning_rate):
       self.value_model.optimizer.lr = learning_rate
       
       for trajectory in self.memory:
           
           st0 = np.zeros((1, config.FEED_MEM, self.input_size))
           st1 = st0
           for state, _, _, next_state, _, _ in trajectory:
               st0 = np.concatenate((st0, np.expand_dims(state, axis = 0)),axis = 0)
               st1 = np.concatenate((st1, np.expand_dims(next_state, axis = 0)), axis = 0)

           st0 = st0[1:]
           st1 = st1[1:]

           target_not_done = self.value_model(st1)
           target_f = np.array(self.value_model(st0))
           
           for i, (_, _, reward, _, _, done) in enumerate(trajectory):     
               target = reward
               if not done:
                   target = reward + self.discount * target_not_done[i]
               target_f[i] = target
           self.value_model.fit(st0, target_f, verbose = 0)
   
   def train_policy_model(self, learning_rate):
       self.policy_model.optimizer.lr = learning_rate
       improvement = 0
       for m, trajectory in enumerate(self.memory):
           st0 = np.zeros((1, config.FEED_MEM, self.input_size))
           st1 = st0
           grad_accumulate = []
           for state, _, _, next_state, grad, _ in trajectory:
               st0 = np.concatenate((st0, np.expand_dims(state, axis = 0)),axis = 0)
               st1 = np.concatenate((st1, np.expand_dims(next_state, axis = 0)), axis = 0)
               grad_accumulate.append(grad)
               
           st0 = st0[1:]
           st1 = st1[1:]
       
           q_value = np.array(self.discount * self.value_model(st1))
           v_value = np.array(self.value_model(st0))
           for i, (_, _, reward, _, _, done) in enumerate(trajectory):
               if done:
                   q_value[i] = reward
               else:
                   q_value[i] += reward 
           
           for i, grad in enumerate(grad_accumulate):
               improvement -= np.array(grad) * (q_value[i] - v_value[i])
           
       improvement /= m
       self.policy_model.optimizer.apply_gradients(zip(improvement, self.policy_model.trainable_weights))
       
       
   def load(self, name):
       self.policy_model.load_weights(name + "policy.hdf5")
       self.value_model.load_weights(name + "value.hdf5")
       
   def save(self, name):
       self.policy_model.save_weights(name + "policy.hdf5")
       self.value_model.save_weights(name + "value.hdf5")
