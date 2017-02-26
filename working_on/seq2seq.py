import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
import matplotlib.pyplot as plt

np.random.seed(7)


def generate_sequences(batch_num, sequence_length, batch_size):
    x_data = np.random.uniform(0, 1, size=(batch_num, sequence_length, batch_size, 1))
    x_data = np.array(x_data, dtype=np.float32)

    y_data = np.random.uniform(0, 1, size=(batch_num, sequence_length, batch_size, 1))
    y_data = np.array(y_data, dtype=np.float32)
    return x_data, y_data

def main():
    batch_size = 200
    sequence_num = batch_size
    sequence_length = 10
    data_point_dim = 1

    inputs, outputs = generate_sequences(batch_num=sequence_num//batch_size, sequence_length=sequence_length,
                                         batch_size=batch_size)
    print("inputs:",inputs.shape)#[batch_num, sequence_length, batch_size, data_point_dim]
    encoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in range(sequence_length)]

    print("encoder_inputs[:",sequence_length,"].get_shape()>>",encoder_inputs[0].get_shape())
    decoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in range(sequence_length)]
    print("decoder_inputs[:",sequence_length,"].get_shape()>>",decoder_inputs[0].get_shape())

    #cell = tf.nn.rnn_cell.LSTMCell(data_point_dim,use_peepholes=True,state_is_tuple=True)
    #
    cell = tf.nn.rnn_cell.BasicLSTMCell(data_point_dim, state_is_tuple=True)
    cell = tf.nn.rnn_cell.MultiRNNCell( [cell] )

    model_outputs, states = seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                                      decoder_inputs,
                                                      cell)

    cost = tf.reduce_mean(tf.squared_difference(model_outputs, decoder_inputs))

    print("cost.get_shape()>>",cost.get_shape())
    step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # writer = tf.train.SummaryWriter("/tmp/tensor/train", session.graph, )

        costs = []
        n_iterations = 2300
        for i in range(n_iterations):
            batch_costs = []
            for batch_inputs, batch_outputs in zip(inputs, outputs):
                #inputs=>[batch_num, sequence_length, batch_size, data_point_dim]
                #print("batch_inputs:",batch_inputs.shape)
                #batch_inputs=>[sequence_length, batch_size, data_point_dim]
                x_list = {key: value for (key, value) in zip(encoder_inputs, batch_inputs)}
                #value=>[batch_size, data_point_dim]
                y_list = {key: value for (key, value) in zip(decoder_inputs, batch_outputs)}
                x_list.update(y_list);
                err, _ = session.run([cost, step],x_list)
                batch_costs.append(err)
            # if summary is not None:
            #     writer.add_summary(summary, i)
            aveErr=np.average(batch_costs, axis=0);
            print("aveErr>",i,">",aveErr)
            if(aveErr<0.0005):break
            costs.append(aveErr)

        plt.plot(costs)
        plt.show()
        for batch_inputs, batch_outputs in zip(inputs, outputs):
            x_list = {key: value for (key, value) in zip(encoder_inputs, batch_inputs)}
            #value=>[batch_size, data_point_dim]
            y_list = {key: value for (key, value) in zip(decoder_inputs, batch_outputs)}
            x_list.update(y_list);
            output= session.run([model_outputs],x_list)
            output=np.array(output)
            for i in range(3):
                print("output:",output.shape)
                plt.plot(inputs[0,:,i,0], label='inputs')
                plt.plot(output[0,:,i,0], label='ref_output')
                batch_outputs=np.array(batch_outputs)
                print("batch_outputs:",batch_outputs.shape)
                plt.plot(batch_outputs[:,i,0], label='batch_outputs')
                plt.legend()
                plt.show()


if __name__ == '__main__':
    main()
