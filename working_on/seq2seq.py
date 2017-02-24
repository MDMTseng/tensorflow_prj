import numpy as np
import tensorflow as tf
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import rnn_cell
import matplotlib.pyplot as plt

np.random.seed(7)


def generate_sequences(batch_num, sequence_length, batch_size):
    x_data = np.random.uniform(0, 1, size=(batch_num, sequence_length, batch_size, 1))
    x_data = np.array(x_data, dtype=np.float32)

    print(x_data.shape)
    y_data = []
    for x in x_data:
        sequence = [x[0]]
        for index in range(1, len(x)):
            sequence.append(x[0] * x[index])
        # sequence.append([np.max(sequence, axis=0)])
        # candidates_for_min = sequence[1:]
        # sequence.append([np.min(candidates_for_min, axis=0)])
        y_data.append(sequence)

    return x_data, y_data

def main():
    sequence_num = 1000
    sequence_length = 100
    batch_size = 1000
    data_point_dim = 1

    inputs, outputs = generate_sequences(batch_num=sequence_num//batch_size, sequence_length=sequence_length,
                                         batch_size=batch_size)
    print("inputs:",inputs.shape)#[batch_num, sequence_length, batch_size, data_point_dim]
    encoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in range(sequence_length)]

    print("encoder_inputs[:",sequence_length,"].get_shape()>>",encoder_inputs[0].get_shape())
    decoder_inputs = [tf.placeholder(tf.float32, shape=[batch_size, data_point_dim]) for _ in range(sequence_length)]
    print("decoder_inputs[:",sequence_length,"].get_shape()>>",decoder_inputs[0].get_shape())

    model_outputs, states = seq2seq.basic_rnn_seq2seq(encoder_inputs,
                                                      decoder_inputs,
                                                      rnn_cell.BasicLSTMCell(data_point_dim, state_is_tuple=True))

    reshaped_outputs = tf.reshape(model_outputs, [-1])
    reshaped_results = tf.reshape(decoder_inputs, [-1])

    cost = tf.reduce_mean(tf.squared_difference(reshaped_outputs, reshaped_results))

    step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cost)

    with tf.Session() as session:
        session.run(tf.global_variables_initializer())
        # writer = tf.train.SummaryWriter("/tmp/tensor/train", session.graph, )

        costs = []
        n_iterations = 300
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
            if(aveErr<0.01):break
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
            print("output:",output.shape)
            plt.plot(output[0,:,0,0])
            batch_outputs=np.array(batch_outputs)
            print("batch_outputs:",batch_outputs.shape)
            plt.plot(batch_outputs[:,0,0])
            plt.show()


if __name__ == '__main__':
    main()
