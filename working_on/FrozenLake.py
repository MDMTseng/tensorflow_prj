
import gym
import numpy as np
import random
import tensorflow as tf


env = gym.make('FrozenLake-v0')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[None,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,4],0,0.01))
B = tf.Variable(tf.random_uniform([4],0,0.01))
Qout = tf.nn.tanh(tf.matmul(inputs1,W)+B)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))


rate_g = tf.train.exponential_decay(0.001, loss, 1, 0.99)
updateModel = tf.train.RMSPropOptimizer(rate_g).minimize(loss)


init = tf.initialize_all_variables()

def printQ(sess,env):
    Qmap = sess.run([Qout],feed_dict={inputs1:np.identity(16)})
    print("Qmap>>#a=0:L, 1:D , 2:R, 3:U")
    print(Qmap)
    env.render()


# Set learning parameters
e = 0.5
num_episodes = 4000
#create lists to contain total rewards and steps per episode
ave_r = 0
with tf.Session() as sess:
    sess.run(init)
    print("env.action_space>>",env.action_space.__dict__)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        ns = env.reset()
        end = False
        j = 0


        #Choose an action by greedily (with e chance of random action) from the Q-network
        [nQ] = sess.run([Qout],feed_dict={inputs1:[np.identity(16)[ns]]})
        #The Q-Network
        while j < 20:
            j+=1
            cs,cQ,a = ns,nQ,np.argmax(nQ)
            #print(cs)

            if np.random.rand(1) < e:
                a = env.action_space.sample()

            
            ns,nr,end,_ = env.step(a)


            if( j == 20 ):
                end=True#enough
            if( end==True and nr!=1 ) :
                nr=-1
            nQ = sess.run(Qout,feed_dict={inputs1:[np.identity(16)[ns]]})
            #Obtain maxQ' and set our target value for chosen action.
            cQ[0,a] += 0.5*( nr + 0.5*(np.max(nQ)+np.min(nQ))/2 - cQ[0,a])
            #Train our network using target and predicted Q values
            _ = sess.run([updateModel],feed_dict={inputs1:[np.identity(16)[cs]],nextQ:cQ})

            '''if (i%500 == 0) :
                env.render()'''

            if end == True:
                e *= 0.9
                if (i%500 == 499) :
                    printQ(sess,env)
                break

        if nr == 1 :ave_r +=1
        if i%100 == 0 :
            print ("episodes: ", i, " ave_r:", ave_r,"%")
            ave_r = 0
