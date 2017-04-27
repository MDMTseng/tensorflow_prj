
import gym
import numpy as np
import random
import tensorflow as tf
from gym.envs.registration import register


env = gym.make('CartPole-v0')
tf.reset_default_graph()


def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         rlx=tf.nn.relu(x);
         rlnx=tf.nn.relu(-x);
         return rlx-leak*rlnx
def relu(x, name="relu"):
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

            W = tf.Variable(tf.random_uniform([input_dim,50],-0.01,0.01))
            B = tf.Variable(tf.random_uniform([50],-0.01,0.01))
            Wo = tf.Variable(tf.random_uniform([50,output_dim],-0.01,0.01))
            Bo = tf.Variable(tf.random_uniform([output_dim],-0.01,0.01))

            H1 = tf.nn.relu(tf.matmul(net_input,W)+B)
            #H2 = tf.nn.relu(tf.matmul(H1,W2)+B2)
            self.output = tf.nn.tanh(tf.matmul(H1,Wo)+Bo)
            self.Ws = [W,B,Wo,Bo];


    def Net_train_graph(self,graph,net_input,targetOutput):
        with graph.as_default():
            loss = tf.reduce_sum(tf.square(targetOutput - self.output))
            L2_reg = [p.assign(p*0.999) for p in self.Ws]
            L1_reg = [p.assign(p-0.000001*tf.sign(p)) for p in self.Ws]

            regularizer = tf.contrib.layers.l2_regularizer(scale=0.01, scope=None)
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


    exp_set={'cs':[],'a':[],'nr':[],'ns':[]}
    def training_exp_replay_set(self,experience=exp_set):
        self.training_exp_replay(experience['cs'],experience['a'],experience['nr'],experience['ns'])
    def training_exp_replay(self,cs_arr,a_arr,nr_arr,ns_arr):
        # current state, action, next reward, next state
        # => current state, Q; next state-> next Q(nQ)
        # => target Q = nr + gamma*(max(nQ))
        # => train (current state, target Q)
        alpha = 0.7
        gamma = 0.3
        with self.graph.as_default():
            [cQ] = self.QNet(cs_arr);
            [nQ] = self.QNet(ns_arr);
            for i in range(len(cQ)):
                #print("cs:",cs_arr[i]," a:",a_arr[i]," nr:",nr_arr[i]," ns:",ns_arr[i],"<<<<")
                cQ[i,a_arr[i]] = cQ[i,a_arr[i]]+alpha*( nr_arr[i] + gamma* (np.max(nQ[i])) -cQ[i,a_arr[i]])

            #for iii in range(len(cQ)):print(cQ[iii], "<><><", np.argmax( cs_arr[iii]))
            sess.run([self.net_obj.train],feed_dict={self.state_in:cs_arr,self.targetQ_in:cQ})

            if(nr_arr[i] !=0):
                nQ.fill(0)

                sess.run([self.net_obj.train],feed_dict={self.state_in:ns_arr,self.targetQ_in:nQ})

    def training(self,states,tarQ_arr):
        with self.graph.as_default():
            sess.run([self.net_obj.train],feed_dict={self.state_in:states,self.targetQ_in:tarQ_arr})

    experience_limit=300
    def experience_append(self,cs,a,nr,ns):

        if len(self.exp_set['cs']) <self.experience_limit:
            self.exp_set['cs'].append(cs)
            self.exp_set[ 'a'].append(a)
            self.exp_set['nr'].append(nr)
            self.exp_set['ns'].append(ns)
        else:
            idx = np.random.randint(len(self.exp_set[ 'cs']))
            self.exp_set[ 'cs'][idx] = cs
            self.exp_set[ 'a'][idx] = a
            self.exp_set[ 'nr'][idx] = nr
            self.exp_set[ 'ns'][idx] = ns

    rewardHist_offset =1
    rewardHist=[0,0,0]
    def experience_append_b(self,cs,a,nr,ns):

        if len(self.exp_set['cs']) <self.experience_limit:
            self.exp_set['cs'].append(cs)
            self.exp_set[ 'a'].append(a)
            self.exp_set['nr'].append(nr)
            self.exp_set['ns'].append(ns)
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
        ns_vec = ns
        [nQ] =  qobj.QNet([ns_vec])
        #The Q-Network
        maxMove = 1200
        nr = 0
        acc_nr =0

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

            exp_x.append({'s':cs_vec,'a':a,'ns':ns_vec})

            if i%1000 == 999 :
                printQ(sess,env,ns_vec,nQ)
                #env.render()
            if end == True:
                ccc=0
                margin = 5
                for exp in exp_x:
                    rw = 0
                    if(ccc > acc_nr-margin):
                        rw=-1
                    elif(ccc < acc_nr/2-margin):
                        rw = 1
                    qobj.experience_append(exp['s'],exp['a'],rw,exp['ns'])
                    ccc+=1

                qobj.training_exp_replay_set()
                break;


        if i%30 == 29 :
            e = (e-0.2)*0.99+0.2
            print ("episodes: ", i, " acc_nr:", acc_nr,"% e:",e, "rewardHist:",qobj.rewardHist, " len(exp):",len(qobj.exp_set['nr']))
            ave_r = 0
