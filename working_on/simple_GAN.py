import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


mb_size = 128
Z_dim = 100

def xavier_init(size):
    in_dim = size[0]
    xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
    return tf.random_normal(shape=size, stddev=xavier_stddev)


def plot(samples):
    fig = plt.figure(figsize=(4, 4))
    gs = gridspec.GridSpec(4, 4)
    gs.update(wspace=0.05, hspace=0.05)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample.reshape(28, 28), cmap='Greys_r')

    return fig

def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

with tf.name_scope('Real_Sample'):
    X = tf.placeholder(tf.float32, shape=[None, 784], name='X')
with tf.name_scope('Gen_Seed'):
    Z = tf.placeholder(tf.float32, shape=[None, Z_dim], name='Z')

with tf.name_scope('G_WS'):
    G_W1 = tf.Variable(xavier_init([Z_dim, 228]), name='G_W1')
    G_b1 = tf.Variable(tf.zeros(shape=[228]), name='G_b1')
    G_W1x = tf.Variable(xavier_init([228, 428]), name='G_W1x')
    G_b1x = tf.Variable(tf.zeros(shape=[428]), name='G_b1x')
    G_W2 = tf.Variable(xavier_init([428, 784]), name='G_W2')
    G_b2 = tf.Variable(tf.zeros(shape=[784]), name='G_b2')
def generator(z,name='generator'):
    with tf.name_scope(name):
        G_h1 = tf.nn.tanh(tf.matmul(z, G_W1) + G_b1)
        G_h2 = tf.nn.tanh(tf.matmul(G_h1, G_W1x) + G_b1x)
        G_log_prob = tf.matmul(G_h2, G_W2) + G_b2
        G_prob = tf.nn.sigmoid(G_log_prob)
        variable_summaries(G_prob)
        return G_prob

def lrelu(x, leak=0.2, name="lrelu"):
     with tf.variable_scope(name):
         rlx=tf.nn.relu(x);
         rlnx=tf.nn.relu(-x);
         return rlx-leak*rlnx

with tf.name_scope('D_WS'):
    D_W1 = tf.Variable(xavier_init([784, 428]), name='D_W1')
    D_b1 = tf.Variable(tf.zeros(shape=[428]), name='D_b1')
    D_W1x = tf.Variable(xavier_init([428, 228]), name='D_W1x')
    D_b1x = tf.Variable(tf.zeros(shape=[228]), name='D_b1x')
    D_W2 = tf.Variable(xavier_init([228, 1]), name='D_W2')
    D_b2 = tf.Variable(tf.zeros(shape=[1]), name='D_b2')
def discriminator(x,name='discriminator'):
    with tf.name_scope(name):
        D_h1 = tf.nn.tanh(tf.matmul(x, D_W1) + D_b1)
        D_h2 = lrelu(tf.matmul(D_h1, D_W1x) + D_b1x)
        D_logit = tf.matmul(D_h2, D_W2) + D_b2
        variable_summaries(D_logit)
        return D_logit



# D_loss = -tf.reduce_mean(tf.log(D_real) + tf.log(1. - D_fake))
# G_loss = -tf.reduce_mean(tf.log(D_fake))

# Alternative losses:
# -------------------

learning_rate = 5e-5*3

with tf.name_scope('GD'):
    G_sample = generator(Z)
    D_logit_fake = discriminator(G_sample,name='D_fake_nn')
    with tf.name_scope('G_loss'):
        #G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.ones_like(D_logit_fake)))
        G_loss =-tf.reduce_mean(D_logit_fake);
        variable_summaries(G_loss)
    rate_g = tf.train.exponential_decay(learning_rate, G_loss, 1, 0.99)
    # Gradient descent
    theta_G = [v for v in tf.global_variables() if "G_WS/G_" in v.name]
    [print(v.name) for v in theta_G]
    G_solver = tf.train.RMSPropOptimizer(rate_g).minimize(G_loss, var_list=theta_G)

with tf.name_scope('D'):
    with tf.name_scope('D_loss'):
        D_logit_real = discriminator(X,name='D_real_nn')
        #D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_real, labels=tf.ones_like(D_logit_real)))
        #D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_logit_fake, labels=tf.zeros_like(D_logit_fake)))
        #D_loss = D_loss_real + D_loss_fake
        D_loss = tf.reduce_mean(D_logit_real) - tf.reduce_mean(D_logit_fake)
    rate_D = tf.train.exponential_decay(learning_rate, D_loss, 1, 0.98)
    theta_D = [v for v in tf.global_variables() if "D_WS/D_" in v.name]
    [print(v.name) for v in theta_D]
    D_solver = tf.train.RMSPropOptimizer(rate_D).minimize(-D_loss, var_list=theta_D)
    # theta_D is list of D's params
    clip_D_WS = [p.assign(tf.clip_by_value(p, -0.1, 0.1)) for p in theta_D]


mnist = input_data.read_data_sets('../../MNIST_data', one_hot=True)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

merged = tf.summary.merge_all()
train_writer = tf.summary.FileWriter('train',sess.graph)
if not os.path.exists('out/'):
    os.makedirs('out/')

i = 0

def sample_Z(m, n):
    return np.random.uniform(-1., 1., size=[m, n])


for it in range(1000000):
    if it % 1000 == 0:
        samples = sess.run(G_sample, feed_dict={Z: sample_Z(16, Z_dim)})
        fig = plot(samples)
        plt.savefig('out/{}.png'.format(str(i).zfill(3)), bbox_inches='tight')
        i += 1
        plt.close(fig)

    X_mb, _ = mnist.train.next_batch(mb_size)


    if it % 1000 == 0:
        _, D_loss_curr,_,merged_res = sess.run([D_solver, D_loss,clip_D_WS,merged], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        train_writer.add_summary(merged_res, it)
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
        print('Iter: {}'.format(it))
        print('D loss: {:.4}'. format(D_loss_curr))
        print('G_loss: {:.4}'.format(G_loss_curr))
        print()
    else:
        _, D_loss_curr,_ = sess.run([D_solver, D_loss,clip_D_WS], feed_dict={X: X_mb, Z: sample_Z(mb_size, Z_dim)})
        _, G_loss_curr = sess.run([G_solver, G_loss], feed_dict={Z: sample_Z(mb_size, Z_dim)})
