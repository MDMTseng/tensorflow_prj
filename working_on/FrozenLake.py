
import gym
import numpy as np
import random
import tensorflow as tf
import matplotlib.pyplot as plt


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


rate_g = tf.train.exponential_decay(0.001, loss, 1, 0.95)
updateModel = tf.train.RMSPropOptimizer(rate_g).minimize(loss)


init = tf.initialize_all_variables()

# Set learning parameters
y = .5
e = 0.1
num_episodes = 4000
#create lists to contain total rewards and steps per episode
jList = []
rList = []
ave_r = 0
with tf.Session() as sess:
    sess.run(init)



    print("env.action_space>>",env.action_space.__dict__)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        s = env.reset()
        rAll = 0
        d = False
        j = 0

        #The Q-Network
        while j < 20:
            j+=1

            #Choose an action by greedily (with e chance of random action) from the Q-network
            inp1=np.identity(16)[s]
            [allQ] = sess.run([Qout],feed_dict={inputs1:[inp1]})

            a = np.argmax(allQ)

            if np.random.rand(1) < e:
                a = env.action_space.sample()

            #print("allQ>>",allQ, " a>>",a, "type(a)>>",type(a))
            #print("a>>",a)
            #Get new state and reward from environment
            s1,r,d,_ = env.step(a)
            ave_r +=r
            if( (d==True or j==19) and r!=1 ) :
                r=-1
            if (i%1000 == 0) :
                env.render()
            #Obtain the Q' values by feeding the new state through our network
            inp1=np.identity(16)[s1]
            Q1 = sess.run(Qout,feed_dict={inputs1:[inp1]})
            #Obtain maxQ' and set our target value for chosen action.
            maxQ1 = np.max(Q1)
            #print("allQ>>",allQ)
            targetQ = allQ

            targetQ[0,a] = r + y*maxQ1
            #Train our network using target and predicted Q values
            _ = sess.run([updateModel],feed_dict={inputs1:np.identity(16)[s:s+1],nextQ:targetQ})
            rAll += r
            s = s1
            if d == True:
                #Reduce chance of random action as we train the model.

                e = 1./((i/50) + 10)
                break

        if i%100 == 0 :
            print ("episodes: ", i, " ave_r:", ave_r/100)
            ave_r = 0
        jList.append(j)
        rList.append(rAll)
print ("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

plt.plot(rList)

plt.plot(jList)
