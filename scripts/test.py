import os
import numpy as np
import tensorflow as tf

#import resultStorer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class ModelTest:

  # Initial weight variable
  def init_weight_variable(self, varName, shape):
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_normal_initializer()
    return tf.get_variable(varName, shape, initializer = initial)
  
  # Initial bias variable
  def init_bias_variable(self, varName, shape):
    #initial = tf.constant(0.1, shape=shape)
    initial = tf.constant_initializer(0.1)
    return tf.get_variable(varName, shape, initializer = initial)

  def buildGraph(self):
    x = tf.constant([[0., 1., 2., 3.], [4., 5., 6., 7.]])
    with tf.variable_scope('foo1'):
      #w_hidden = self.init_weight_variable("w_hidden1", [4, 3])
      w_hidden = tf.constant([[1., 2., 3.], [1., 2., 3.], [1., 2., 3.], [1., 2., 3.,]])
      #b_hidden = self.init_bias_variable("b_hidden1", [3])
      b_hidden = tf.constant([[1., 1., 1.]])
      #h_hidden = tf.nn.relu(tf.matmul(x, w_hidden) + b_hidden)
      h_hidden = tf.matmul(x, w_hidden) + b_hidden
    with tf.variable_scope('foo2'):
      #w_hidden = tf.get_variable("w_hidden2", [3, 2])
      w_hidden = tf.constant([[1., 2.], [1., 2.], [1., 2.]])
      #b_hidden = self.init_bias_variable("b_hidden2", [2])
      b_hidden = tf.constant([[1., 1.]])
      h_hidden = tf.matmul(h_hidden, w_hidden) + b_hidden
    with tf.variable_scope('foo3'):
      #w_hidden = tf.get_variable("w_hidden3", [2, 1])
      w_hidden = tf.constant([[1.], [1.]])
      #b_hidden = self.init_bias_variable("b_hidden3", [1])
      b_hidden = tf.constant([[1.]])
      #self.h_hidden = tf.nn.relu(tf.matmul(h_hidden, w_hidden) + b_hidden)
      self.h_hidden = tf.matmul(h_hidden, w_hidden) + b_hidden
    self.init = tf.global_variables_initializer()

  def trainer(self):
    with tf.Session() as sess:
      sess.run(self.init)
      with tf.variable_scope('foo3'):
        res = sess.run(self.h_hidden)
        print res 
      #with tf.variable_scope('foo2'):
      #  res = sess.run(self.h_hidden)
      #  print res 

def fuc():
  a = np.array([1, 2, 3, 4, 5])
  for ind, ele in enumerate(a):
    print ele
    if ele == 4:
      a = np.delete(a, ind, axis = 0)
    print a

if __name__ == "__main__":
  #x = tf.constant([[0., 1., 2., 3.], [4., 5., 6., 7.]])
  #with tf.variable_scope('foo1'):
  #  w_hidden = init_weight_variable("w_hidden1", [4, 3])
  #  b_hidden = init_bias_variable("b_hidden1", [3])
  #  h_hidden = tf.nn.relu(tf.matmul(x, w_hidden) + b_hidden)
  #with tf.variable_scope('foo2'):
  #  w_hidden = tf.get_variable("w_hidden2", [3, 2])
  #  b_hidden = init_bias_variable("b_hidden2", [2])
  #  h_hidden = tf.nn.relu(tf.matmul(h_hidden, w_hidden) + b_hidden)
  
  #with tf.Session() as sess:
  #  init = tf.global_variables_initializer()
  #  sess.run(init)
  #  res = sess.run(h_hidden)
  #  print res
    #with tf.variable_scope('foo', reuse=True):
    #  w_hidden2 = tf.get_variable("w_hidden1")
    #  print sess.run(tf.get_variable("w_hidden1"))
    #  print sess.run(w_hidden2)
  fuc()
  raw_input("..")

  a = input("...")
  print type(a)
  
  a = raw_input("...")
  ins = ModelTest()
  ins.buildGraph()
  ins.trainer()
