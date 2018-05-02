
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

# Parameters
learning_rate = 0.001
batch_size = 200
display_step = 1

# Network Parameters
n_hidden_1 = 256 # 1st layer number of features
n_hidden_2 = 256 # 2nd layer number of features
n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)





# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}


def MNIST_CN_GetNetwork():

    # tf Graph input
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    # Construct model
    pred = multilayer_perceptron(x, weights, biases)
    #Add L2 reg
    l2_Loss = tf.nn.l2_loss(weights['h1']) + tf.nn.l2_loss(weights['h2']) + tf.nn.l2_loss(weights['out']) + \
                 tf.nn.l2_loss(biases['b1']) + tf.nn.l2_loss(biases['b2']) + tf.nn.l2_loss(biases['out'])
    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y)+0.01*l2_Loss)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    return x,y,pred,cost,optimizer




def MNIST_CN_Traning(x,y,pred,cost,optimizer,epoch_count,model_import_path,model_export_path):
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    # Running first session
    print("Starting modelTraning...")
    print("epoch_count:",epoch_count)
    print("model_import_path:",model_import_path)
    print("model_export_path:",model_export_path)
    with tf.Session() as sess:

        # Run the initializer
        sess.run(init)
        if model_import_path is not None:
            saver.restore(sess, model_import_path)
        # Training cycle
        for epoch in range(epoch_count):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                              y: batch_y})
                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        Accuracy=accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
        print("Accuracy:", Accuracy)

        if model_export_path is not None:
            # Save model weights to disk
            save_path = saver.save(sess, model_export_path)
            print("Model saved in file: %s" % save_path)
        return Accuracy
