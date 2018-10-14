#import dependencies
from __future__ import print_function
import gym
import numpy as np
import tensorflow as tf
import random

#load the environment
env = gym.make('FrozenLake-v0')
s = env.reset()
print(s)

#Q-network implementation

tf.reset_default_graph()
inputs = tf.placeholder(shape=[None,env.observation_space.n], dtype=tf.float32)

W = tf.get_variable(name="W", dtype=tf.float32, shape=[env.observation_space.n,env.action_space.n],
    initializer = tf.contrib.layers.xavier_initializer())

b = tf.Variable(tf.zeros(shape=[env.action_space.n]), dtype=tf.float32)

qpred = tf.add(tf.matmul(inputs,W),b)
apred = tf.argmax(qpred,1)

qtar = tf.placeholder(shape=[1,env.action_space.n], dtype=tf.float32)
loss = tf.reduce_sum(tf.square(qtar - qpred))

train = tf.train.AdamOptimizer(learning_rate=0.001)
minimizer = train.minimize(loss)

#training the neural network
init = tf.global_variables_initializer()

#setting parameters
y = 0.9
e = 0.4
episodes = 100000

with tf.Session() as sess:
    sess.run(init)

    for i in range(episodes):
        s = env.reset()
        r_total = 0

        while(True):
            a_pred,q_pred = sess.run([apred,qpred], feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1]})

            if np.random.uniform(low=0,high=1)<e:
                a_pred[0] = env.action_space.sample()
            s_,r,t,_ = env.step(a_pred[0])

            if r==0:
                if t==True:
                    r = -10
                else:
                    r = -1

            if r==1:
                r = 10
            q_pred_new = sess.run(qpred, feed_dict={inputs:np.identity(env.observation_space.n)[s_:s_ + 1]})

            targetQ = q_pred
            max_qpredn = np.max(q_pred_new)

            targetQ[0,a_pred[0]] = r + y*max_qpredn

            _ = sess.run(minimizer, feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1], qtar:targetQ})
            s = s_
            if t==True:
                break

    #Testing the performance
    s = env.reset()

    while(True):
        a = sess.run(apred, feed_dict={inputs:np.identity(env.observation_space.n)[s:s+1]})
        s_,r,t,_ = env.step(a[0])
        print("##################################")
        env.render()
        s = s_
        if t==True:
            break
