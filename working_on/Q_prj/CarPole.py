
import gym
import numpy as np
import random
import tensorflow as tf
from gym.envs.registration import register


env = gym.make('CartPole-v0')
tf.reset_default_graph()


def lrelu(x, leak=0.1, name="lrelu"):
     rlx=leak*x + (1-leak)*tf.nn.relu(x);
     return rlx
def relu(x, name="relu"):
    with tf.variable_scope(name):
        return tf.nn.relu(x)
def tanh(x):
       return tf.nn.tanh(x)



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
        input_dim = net_input.shape.as_list()[1];
        output_dim = targetOutput.shape.as_list()[1];

        W = tf.Variable(tf.random_uniform([input_dim,50],-0.001,0.001))
        B = tf.Variable(tf.random_uniform([50],-0.001,0.001))


        Wo = tf.Variable(tf.random_uniform([50,output_dim],-0.001,0.001))
        Bo = tf.Variable(tf.random_uniform([output_dim],-0.001,0.001))

        H1_val = tf.matmul(net_input,W)+B

        H1 = lrelu(H1_val)
        #H2 = tf.nn.relu(tf.matmul(H1,W2)+B2)
        output = (tf.matmul(H1,Wo)+Bo)

        self.output = tanh(output)
        self.Ws = [W,B,Wo,Bo];



    def Net_train_graph(self,graph,net_input,targetOutput):
        loss = tf.reduce_sum(tf.square(targetOutput - self.output))
        L2_reg = [p.assign(p*0.999) for p in self.Ws]
        L1_reg = [p.assign(p-0.000001*tf.sign(p)) for p in self.Ws]

        regularizer = tf.contrib.layers.l2_regularizer(scale=0.001, scope=None)
        regularization_penalty = tf.contrib.layers.apply_regularization(regularizer, self.Ws)

        regularized_loss = loss +regularization_penalty  # this loss needs to be minimized

        self.train = tf.train.AdamOptimizer(0.001).minimize(regularized_loss)


class QLearning_OBJ:
    net_obj = None
    targetQ_in = None
    state_in = None
    net_obj = None
    graph = None


    def __init__(self,graph,inDim,OutDim):
        self.graph = graph
        with self.graph.as_default():

            self.state_in = tf.placeholder(shape=[None,inDim],dtype=tf.float32)
            self.targetQ_in = tf.placeholder(shape=[None,OutDim],dtype=tf.float32)
            self.net_obj = NNetwork_Obj(graph1,self.state_in, self.targetQ_in)

    def QNet(self,states):
        with self.graph.as_default():
            return sess.run([self.net_obj.output],feed_dict={self.state_in:states})


    exp_set={'cs':[],'a':[],'nr':[],'ns':[],'isEnd':[]}
    def training_exp_replay_set(self,experience=exp_set):
        self.training_exp_replay(experience['cs'],experience['a'],experience['nr'],experience['ns'],experience['isEnd'])
    def training_exp_replay(self,cs_arr,a_arr,nr_arr,ns_arr,isEnd_arr):
        # current state, action, next reward, next state
        # => current state, Q; next state-> next Q(nQ)
        # => target Q = nr + gamma*(max(nQ))
        # => train (current state, target Q)
        alpha = 0.7
        gamma = 0.9
        with self.graph.as_default():
            [cQ] = self.QNet(cs_arr);
            [nQ] = self.QNet(ns_arr);
            for i in range(len(cQ)):
                #print("cs:",cs_arr[i]," a:",a_arr[i]," nr:",nr_arr[i]," ns:",ns_arr[i],"<<<<")
                if(isEnd_arr[i]):
                    tmp_cQ = nr_arr[i]
                else:
                    tmp_cQ = nr_arr[i] + gamma* np.max(nQ[i])

                if(tmp_cQ>1):tmp_cQ=1
                if(tmp_cQ<-1):tmp_cQ=-1
                cQ[i,a_arr[i]]=tmp_cQ

            #for iii in range(len(cQ)):print(cQ[iii], "<><><", np.argmax( cs_arr[iii]))
            sess.run([self.net_obj.train],feed_dict={self.state_in:cs_arr,self.targetQ_in:cQ})


    def training(self,states,tarQ_arr):
        with self.graph.as_default():
            sess.run([self.net_obj.train],feed_dict={self.state_in:states,self.targetQ_in:tarQ_arr})

    def experience_append(self,cs,a,nr,ns,isEnd=False,experience_limit=500):
        if np.random.rand(1) < 0.1:
            return
        if len(self.exp_set['cs']) <self.experience_limit:
            self.exp_set['cs'].append(cs)
            self.exp_set[ 'a'].append(a)
            self.exp_set['nr'].append(nr)
            self.exp_set['ns'].append(ns)
            self.exp_set['isEnd'].append(isEnd)
        else:
            idx = np.random.randint(len(self.exp_set[ 'cs']))
            self.exp_set[ 'cs'][idx] = cs
            self.exp_set[ 'a'][idx] = a
            self.exp_set[ 'nr'][idx] = nr
            self.exp_set[ 'ns'][idx] = ns
            self.exp_set['isEnd'][idx]=isEnd

    rewardHist_offset =1
    rewardHist=[0,0,0]
    def experience_append_b(self,cs,a,nr,ns,isEnd=False,experience_limit=500):

        if len(self.exp_set['cs']) <self.experience_limit:
            self.exp_set['cs'].append(cs)
            self.exp_set[ 'a'].append(a)
            self.exp_set['nr'].append(nr)
            self.exp_set['ns'].append(ns)
            self.exp_set['isEnd'].append(isEnd)
        else:
            most_r = np.argmax( self.rewardHist) - self.rewardHist_offset#find the exp that we had most

            self.rewardHist[int(most_r+self.rewardHist_offset)]-=1
            idx = np.random.randint(len(self.exp_set[ 'cs']))
            while self.exp_set[ 'nr'][idx] != most_r:
                idx = np.random.randint(len(self.exp_set[ 'cs']))

            self.exp_set[ 'cs'][idx] = cs
            self.exp_set[ 'a'][idx] = a
            self.exp_set[ 'nr'][idx] = nr
            self.exp_set[ 'ns'][idx] = ns
            self.exp_set['isEnd'][idx]=isEnd


        self.rewardHist[int(nr+self.rewardHist_offset)]+=1


graph1 = tf.Graph()

print("env.action_space>>",env.action_space.__dict__)
print("env.observation_space>>",env.observation_space.low.shape[0])
qobj = QLearning_OBJ(graph1,env.observation_space.low.shape[0],env.action_space.n)

with graph1.as_default():
    init = tf.initialize_all_variables()




act_map = ['<','>']
def printQ(sess,env,state,Q):
    #[position of cart, velocity of cart, angle of pole, rotation rate of pole]
    print(state)
    print(Q)
    print(act_map[np.argmax( Q)])

# Set learning parameters
e = 0.0
num_episodes = 40000
#create lists to contain total rewards and steps per episode
ave_r = 0





with tf.Session(graph=graph1) as sess:
    sess.run(init)
    print("env.action_space>>",env.action_space.__dict__)

    acc_nr =0
    for i in range(num_episodes):
        #Reset environment and get first new observation
        ns = env.reset()
        end = False
        j = 0


        #Choose an action by greedily (with e chance of random action) from the Q-network
        ns_vec = ns
        [nQ] =  qobj.QNet([ns_vec])
        #The Q-Network
        maxMove = 200
        nr = 0

        exp_x=[]
        while j < maxMove:
            j+=1
            cs,cQ,a,cr = ns,nQ,np.argmax(nQ),nr
            cs_vec = ns_vec
            #print(cs)
            if np.random.rand(1) < e:
                a = env.action_space.sample()

            ns,nr,end,_ = env.step(a)

            acc_nr +=1

            ns_vec = ns
            [nQ] =  qobj.QNet([ns_vec])

            '''if i%100 == 99 :
                printQ(sess,env,ns_vec,nQ)
                #env.render()'''

            if end == False:#reduce reward value to prevent
                qobj.experience_append(cs_vec,a,0.05,ns_vec)
            else:
                turn_lifeTime=j
                ccc=0
                margin = 1
                qobj.experience_append(cs_vec,a,-0.05,ns_vec,True)

                qobj.training_exp_replay_set()
                break;

        summeryC=100
        if i%summeryC == (summeryC-1) :
            e = (e-0.0)*0.99+0.0
            print ("episodes: ", i, " acc_nr:", "%3.2f"% (100.0*acc_nr/(maxMove*summeryC)),"% e:",e)
            ave_r = 0
            acc_nr =0
