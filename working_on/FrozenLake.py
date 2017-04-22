
import gym
import numpy as np
import random
import tensorflow as tf
from gym.envs.registration import register

register(
        id='FrozenLake-v1',
        entry_point='gym.envs.toy_text:FrozenLakeEnv',
        kwargs={'map_name':'4x4',
            'is_slippery':False
            }
        )

env = gym.make('FrozenLake-v1')
tf.reset_default_graph()


def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         rlx=tf.nn.relu(x);
         rlnx=tf.nn.relu(-x);
         return rlx-leak*rlnx
def relu(x, leak=0.2, name="relu"):
    with tf.variable_scope(name):
        return tf.nn.relu(x)



class NNetwork_Obj:
    train = None
    output = None
    Ws = None
    graph = None
    def __init__(self,graph,net_input,targetOutput):
        self.init(graph,net_input,targetOutput)


    def init(self,graph,net_input,targetOutput):
        self.graph=graph
        self.Net_graph(graph,net_input,targetOutput)
        self.Net_train_graph(graph,net_input,targetOutput)

    def Net_graph(self,graph,net_input,targetOutput):
        with graph.as_default():
            input_dim = net_input.shape.as_list()[1];
            output_dim = targetOutput.shape.as_list()[1];

            W = tf.Variable(tf.random_uniform([input_dim,15],-0.01,0.01))
            B = tf.Variable(tf.random_uniform([15],-0.01,0.01))
            Wo = tf.Variable(tf.random_uniform([15,output_dim],-0.01,0.01))
            Bo = tf.Variable(tf.random_uniform([output_dim],-0.01,0.01))

            H1 = relu(tf.matmul(net_input,W)+B)
            #H2 = tf.nn.relu(tf.matmul(H1,W2)+B2)
            self.output = tf.nn.tanh(tf.matmul(H1,Wo)+Bo)
            self.Ws = [W,B,Wo,Bo];


    def Net_train_graph(self,graph,net_input,targetOutput):
        with graph.as_default():
            loss = tf.reduce_sum(tf.square(targetOutput - self.output))
            L2_reg = [p.assign(p*0.999) for p in self.Ws]
            L1_reg = [p.assign(p-0.000001*tf.sign(p)) for p in self.Ws]

            regularizer = tf.contrib.layers.l2_regularizer(scale=0.003, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, self.Ws)

            regularized_loss = loss +regularization_penalty  # this loss needs to be minimized

            self.train = tf.train.RMSPropOptimizer(0.001).minimize(regularized_loss)


class QLearning_OBJ:
    net_obj = None
    targetQ_in = None
    state_in = None
    net_obj = None
    graph = None

    exp_set={'cs':[],'a':[],'nr':[],'ns':[]}
    exp_reward_sum = 0

    def __init__(self,graph):
        self.graph = graph
        with self.graph.as_default():

            self.state_in = tf.placeholder(shape=[None,16],dtype=tf.float32)
            self.targetQ_in = tf.placeholder(shape=[None,4],dtype=tf.float32)
            self.net_obj = NNetwork_Obj(graph1,self.state_in, self.targetQ_in)

    def QNet(self,states):
        with self.graph.as_default():
            return sess.run([self.net_obj.output],feed_dict={self.state_in:states})


    def training_exp_replay_set(self,experience):
        self.training_exp_replay(experience['cs'],experience['a'],experience['nr'],experience['ns'])
    ccc = 0
    def training_exp_replay(self,cs_arr,a_arr,nr_arr,ns_arr):
        # current state, action, next reward, next state
        # => current state, Q; next state-> next Q(nQ)
        # => target Q = nr + gamma*(max(nQ))
        # => train (current state, target Q)
        self.ccc+=1
        with self.graph.as_default():
            [cQ] = self.QNet(cs_arr);
            [nQ] = self.QNet(ns_arr);
            for i in range(len(cQ)):
                #print("cs:",cs_arr[i]," a:",a_arr[i]," nr:",nr_arr[i]," ns:",ns_arr[i],"<<<<")
                cQ[i,a_arr[i]] = nr_arr[i] + 0.5* (np.max(nQ[i]))

            sess.run([self.net_obj.train],feed_dict={self.state_in:cs_arr,self.targetQ_in:cQ})

            if(nr_arr[i] !=0):
                for i in range(len(nQ)):
                    for j in range(len(nQ[i])):
                        nQ[i,j]=0
                sess.run([self.net_obj.train],feed_dict={self.state_in:ns_arr,self.targetQ_in:nQ})
    '''def training_exp_replay(self,cs_arr,a_arr,nr_arr,ns_arr):
        # current state, action, next reward, next state
        # => current state, Q; next state-> next Q(nQ)
        # => target Q = nr + gamma*(max(nQ))
        # => train (current state, target Q)
        self.ccc+=1
        with self.graph.as_default():
            [cQ] = self.QNet(cs_arr);
            [nQ] = self.QNet(ns_arr);
            for i in range(len(cQ)):
                #if nr_arr[i]==1:
                #print("cs:",cs_arr[i]," a:",a_arr[i]," nr:",nr_arr[i]," ns:",ns_arr[i],"<<<<")
                cQ[i,a_arr[i]] = nr_arr[i] + 0.95* (np.max(nQ[i]))
                if( cQ[i,a_arr[i]]< -1 ):
                    cQ[i,a_arr[i]] = -1
            if self.ccc%100 ==0:
                for i in range(len(cQ)):
                    print("cs:",np.argmax( cs_arr[i])," cQ:",cQ[i])
            sess.run([self.net_obj.train],feed_dict={self.state_in:cs_arr,self.targetQ_in:cQ})
            sess.run([self.net_obj.train],feed_dict={self.state_in:cs_arr,self.targetQ_in:cQ})'''

    def training(self,states,tarQ_arr):
        with self.graph.as_default():
            sess.run([self.net_obj.train],feed_dict={self.state_in:states,self.targetQ_in:tarQ_arr})



    def experience_append(self,cs,a,nr,ns):
        self.exp_reward_sum += nr
        if len(self.exp_set['cs']) <500:
            self.exp_set['cs'].append(cs)
            self.exp_set[ 'a'].append(a)
            self.exp_set['nr'].append(nr)
            self.exp_set['ns'].append(ns)
        else:
            idx = 0
            idx = np.random.randint(len(self.exp_set[ 'cs']))
            while self.exp_set[ 'nr'][idx] == 1:
                if np.random.rand(1) <0.1:
                    break
                idx = np.random.randint(len(self.exp_set[ 'cs']))


            self.exp_reward_sum -=self.exp_set[ 'nr'][idx]
            self.exp_set[ 'cs'][idx] = cs
            self.exp_set[ 'a'][idx] = a
            self.exp_set[ 'nr'][idx] = nr
            self.exp_set[ 'ns'][idx] = ns



graph1 = tf.Graph()

qobj = QLearning_OBJ(graph1)

with graph1.as_default():
    init = tf.initialize_all_variables()




def printQ(sess,env):
    Qmap = qobj.QNet(np.identity(16))
    act_map = ['<','V','>','^'];
    print("Qmap>> a=",act_map)
    print(Qmap[0])
    print(np.array([act_map[np.argmax( Qrow)] for Qrow in Qmap[0]]).reshape(4, 4))
    env.render()

# Set learning parameters
e = 0.7
num_episodes = 40000
#create lists to contain total rewards and steps per episode
ave_r = 0





with tf.Session(graph=graph1) as sess:
    sess.run(init)
    print("env.action_space>>",env.action_space.__dict__)
    for i in range(num_episodes):
        #Reset environment and get first new observation
        ns = env.reset()
        end = False
        j = 0


        #Choose an action by greedily (with e chance of random action) from the Q-network
        ns_vec = np.identity(16)[ns]
        [nQ] =  qobj.QNet([ns_vec])
        #The Q-Network
        maxMove = 30
        nr = 0
        while j < maxMove:
            j+=1
            cs,cQ,a,cr = ns,nQ,np.argmax(nQ),nr
            cs_vec = ns_vec
            #print(cs)
            if np.random.rand(1) < e:
                a = env.action_space.sample()


            ns,nr,end,_ = env.step(a)
            ns_vec = np.identity(16)[ns]

            if( j==maxMove or cs==ns) :
                end=True
            if( end==True and nr!=1 ) :
                nr=-1
            [nQ] =  qobj.QNet([ns_vec])

            if np.random.rand(1) <0.1:
                qobj.experience_append(cs_vec,a,nr,ns_vec)

            if end == True:
                qobj.experience_append(cs_vec,a,nr,ns_vec)
                if nr == 1 :ave_r +=1
                qobj.training_exp_replay_set(qobj.exp_set)
                break

        if i%100 == 99 :
            e *= 0.9
            printQ(sess,env)
            print ("episodes: ", i, " ave_r:", ave_r,"% e:",e, " exp_reward_sum:",qobj.exp_reward_sum," len(exp):",len(qobj.exp_set['nr']))
            ave_r = 0
