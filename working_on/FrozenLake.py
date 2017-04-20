
import gym
import numpy as np
import random
import tensorflow as tf
from gym.envs.registration import register

register(
        id='Frozenlake-v3',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name':'4x4',
            'is_slippery':False
            }
        )

env = gym.make('Frozenlake-v3')
tf.reset_default_graph()


def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         rlx=tf.nn.relu(x);
         rlnx=tf.nn.relu(-x);
         return rlx-leak*rlnx

#These lines establish the feed-forward part of the network used to choose actions
state = tf.placeholder(shape=[None,16],dtype=tf.float32)
W = tf.Variable(tf.random_uniform([16,50],-0.01,0.01))
B = tf.Variable(tf.random_uniform([50],-0.01,0.01))
Wo = tf.Variable(tf.random_uniform([50,4],-0.01,0.01))
Bo = tf.Variable(tf.random_uniform([4],-0.01,0.01))

Ws = [W,B,Wo,Bo];
L2_reg = [p.assign(p*0.999) for p in Ws]
L1_reg = [p.assign(p-0.000001*tf.sign(p)) for p in Ws]

H1 = lrelu(tf.matmul(state,W)+B)
#H2 = tf.nn.relu(tf.matmul(H1,W2)+B2)
Qout = lrelu(tf.matmul(H1,Wo)+Bo)

#Below we obtain the loss by taking the sum of squares difference between the target and prediction Q values.
targetQ = tf.placeholder(shape=[None,4],dtype=tf.float32)
loss = tf.reduce_sum(tf.square(targetQ - Qout))




l1_regularizer = tf.contrib.layers.l1_regularizer(scale=0.0001, scope=None)
regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, Ws)

regularized_loss = loss +regularization_penalty # this loss needs to be minimized

updateModel = tf.train.RMSPropOptimizer(0.001).minimize(regularized_loss)


init = tf.initialize_all_variables()

def printQ(sess,env):
    Qmap = sess.run([Qout],feed_dict={state:np.identity(16)})
    act_map = ['<','V','>','^'];
    print("Qmap>> a=",act_map)
    print(Qmap[0])
    print(np.array([act_map[np.argmax( Qrow)] for Qrow in Qmap[0]]).reshape(4, 4))
    env.render()


class Network_Obj:
    def init(input)
    def __init__(self,):


class QLearning_OBJ:
    net_obj = null
    def __init__(self,net_obj):
        self.net_obj = net_obj



        

# Set learning parameters
e = 0.7
num_episodes = 40000
#create lists to contain total rewards and steps per episode
ave_r = 0
exp_set={'cs':[],'a':[],'nr':[],'ns':[]};
with tf.Session() as sess:
    sess.run(init)
    print("env.action_space>>",env.action_space.__dict__)
    inS1_arr=[];
    tarQ_arr=[];
    for i in range(num_episodes):
        #Reset environment and get first new observation
        ns = env.reset()
        end = False
        j = 0


        #Choose an action by greedily (with e chance of random action) from the Q-network
        [nQ] = sess.run([Qout],feed_dict={state:[np.identity(16)[ns]]})
        #The Q-Network
        maxMove = 30
        nr = 0
        while j < maxMove:
            j+=1
            cs,cQ,a,cr = ns,nQ,np.argmax(nQ),nr
            #print(cs)

            if np.random.rand(1) < e:
                a = env.action_space.sample()
            ns,nr,end,_ = env.step(a)

            if( j==maxMove and ns==cs ) :
                end=True
            if( end==True and nr!=1 ) :
                nr=-1
            nQ = sess.run(Qout,feed_dict={state:[np.identity(16)[ns]]})


            #Obtain maxQ' and set our target value for chosen action.
            cQ[0,a] = nr + 0.7*(np.max(nQ))
            #Train our network using target and predicted Q values
            inS1_arr.append(np.identity(16)[cs])
            tarQ_arr.append(cQ[0])

            if end == True:
                exp_set['cs'].append(cs)
                exp_set[ 'a'].append(a)
                exp_set['nr'].append(cr)
                exp_set['ns'].append(ns)
                if nr == 1 :ave_r +=1
                break

        if i%100 == 99 :
            _ = sess.run([updateModel],feed_dict={state:inS1_arr,targetQ:tarQ_arr})
            #sess.run([L1_reg])
            inS1_arr=[]
            tarQ_arr=[]
            e *= 0.95
            printQ(sess,env)
            print ("episodes: ", i, " ave_r:", ave_r/10,"% e:",e)
            ave_r = 0
