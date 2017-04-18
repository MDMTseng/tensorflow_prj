
import gym
import numpy as np
import random
import tensorflow as tf


env = gym.make('FrozenLake-v0')
tf.reset_default_graph()

#These lines establish the feed-forward part of the network used to choose actions
inputs1 = tf.placeholder(shape=[None,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,10],-0.1,0.1))
B = tf.Variable(tf.random_uniform([10],0,0.1))
Wo = tf.Variable(tf.random_uniform([10,4],-0.1,0.1))
Bo = tf.Variable(tf.random_uniform([4],0,0.1))

Ws = [W,B,Wo,Bo];
L2_reg = [p.assign(p*0.999) for p in Ws]
L1_reg = [p.assign(p-0.0001*tf.sign(p)) for p in Ws]

H1 = tf.nn.tanh(tf.matmul(inputs1,W)+B)
#H2 = tf.nn.relu(tf.matmul(H1,W2)+B2)
Qout = tf.nn.tanh(tf.matmul(H1,Wo)+Bo)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
nextQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(nextQ - Qout))


updateModel = tf.train.AdamOptimizer(0.001).minimize(loss)


init = tf.initialize_all_variables()

def printQ(sess,env):
    Qmap = sess.run([Qout],feed_dict={inputs1:np.identity(16)})
    act_map = ['<','V','>','^'];
    print("Qmap>> a=",act_map)
    print(Qmap[0])
    print(np.array([act_map[np.argmax( Qrow)] for Qrow in Qmap[0]]).reshape(4, 4))
    env.render()


# Set learning parameters
e = 0.8
num_episodes = 40000
#create lists to contain total rewards and steps per episode
ave_r = 0
with tf.Session() as sess:
    sess.run(init)
    print("env.action_space>>",env.action_space.__dict__)
    inS1_arr=[];
    nextQ_arr=[];
    for i in range(num_episodes):
        #Reset environment and get first new observation
        ns = env.reset()
        end = False
        j = 0


        #Choose an action by greedily (with e chance of random action) from the Q-network
        [nQ] = sess.run([Qout],feed_dict={inputs1:[np.identity(16)[ns]]})
        #The Q-Network
        maxMove = 99

        while j < maxMove:
            j+=1
            cs,cQ,a = ns,nQ,np.argmax(nQ)
            #print(cs)

            if np.random.rand(1) < e:
                a = env.action_space.sample()

            ns,nr,end,_ = env.step(a)

            '''if( j == maxMove or ns == cs):
                end=True#enough'''
            if( end==True and nr!=1 ) :
                nr=-1
            nQ = sess.run(Qout,feed_dict={inputs1:[np.identity(16)[ns]]})
            #Obtain maxQ' and set our target value for chosen action.
            cQ[0,a] += ( nr + 0.8*(np.max(nQ)) - cQ[0,a])
            #Train our network using target and predicted Q values
            inS1_arr.append(np.identity(16)[cs])
            nextQ_arr.append(cQ[0])

            '''if (i%500 == 0) :
                env.render()'''

            if end == True:
                if nr == 1 :ave_r +=1
                break

        if i%1000 == 999 :
            _ = sess.run([updateModel],feed_dict={inputs1:inS1_arr,nextQ:nextQ_arr})
            sess.run([L2_reg,L1_reg])
            inS1_arr=[]
            nextQ_arr=[]
            e *= 0.99
            printQ(sess,env)
            print ("episodes: ", i, " ave_r:", ave_r,"% e:",e/10)
            ave_r = 0
