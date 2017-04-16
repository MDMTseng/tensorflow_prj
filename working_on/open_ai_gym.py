import tensorflow as tf
import gym

stddev = 1.0
render = True
render_skip = 5
monitor = False

best_weights = tf.Variable(tf.truncated_normal(shape=[4, 1]))
current_weights = tf.Variable(best_weights.initialized_value())

recalculate_current = tf.assign(current_weights, tf.add(best_weights, tf.random_normal(shape=[4, 1], stddev=stddev)))
set_best = tf.assign(best_weights, current_weights)

x = tf.placeholder(tf.float32, shape=[None, 4])
y = tf.cast(tf.less_equal(0.0, tf.matmul(x, current_weights)), tf.int32)

env = gym.make('CartPole-v0')
if monitor:
  env = gym.wrappers.Monitor(env, '/tmp/cartpole-experiment-1', force=True)
observation = env.reset()
if render:
  env.render()
render_counter = 0
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())

  best = 0
  current = 0
  while True:
    action = sess.run(y, feed_dict={x: [observation]})[0][0]
    observation, reward, done, info = env.step(action)
    current += reward

    if render and (render_counter%50 == 0) :
      env.render()

    if done:
      render_counter += 1
      if current >= best:
        best = current
        sess.run(set_best)
        print ('new best: ' + str(best))
      current = 0

      sess.run(recalculate_current)
      observation = env.reset()
