#coding=utf-8

import os
import math
import numpy as np
import tensorflow as tf

from commonModelFunc import *

class BaseCNNModel(CommonModelFunc):

  def __init__(self, FLAGS, insDataPro):
    self.FLAGS = FLAGS
    self.insDataPro = insDataPro

  # Building CNN graph for base model
  def buildBaseCNNModelGraph(self):
    with tf.device("/gpu:0"):
      self.keepProb = tf.placeholder(tf.float32, name = "keepProb")
      self.init = tf.global_variables_initializer()

      with tf.variable_scope("inputLayer"):
        num4Features = self.FLAGS.num4Features
        self.xData = tf.placeholder(tf.float32,
            [self.FLAGS.batchSize,
             num4Features,
             self.FLAGS.maxInputChannels],
            name = "xData")

        self.xInput = tf.reshape(self.xData,
            [-1,
             self.FLAGS.batchSize,
             num4Features,
             self.FLAGS.maxInputChannels],
            name = "xInput")

        self.yLabel = tf.placeholder(tf.float32,
            [1, 2],
            name = "yLabel")

      # First convolutional layer
      with tf.variable_scope("conv1Layer"):
        conv1KHeight = 1
        conv1KWidth = self.FLAGS.conv1KWidth
        conv1SHeight = 1
        conv1SWidth = self.FLAGS.conv1SWidth

        num4InputChannels = self.FLAGS.maxInputChannels
        num4OutputChannels = self.FLAGS.num4OutputChannels

        wConv1 = self.init_weight_variable("wConv1",
            [conv1KHeight,
             conv1KWidth,
             num4InputChannels,
             num4OutputChannels])

        bConv1 = self.init_bias_variable("bConv1", [num4OutputChannels])

        hConv1 = tf.nn.relu(
            self.conv2d(self.xInput,
                wConv1,
                conv1SHeight,
                conv1SWidth,
                num4InputChannels) + bConv1,
            name = "hConv1")

      # ROI polling layer
      with tf.variable_scope("roiPoolingLayer"):
        num4FirstFC = self.FLAGS.num4FirstFC
        shape4hConv1 = hConv1.get_shape().as_list()
        #print shape4hConv1, "......"
        #print num4FirstFC
        num4EachFM = num4FirstFC / num4OutputChannels  # 先保证每个FM贡献同等数量的池化特征；
        #print "num4EachFM:", num4EachFM
        #raw_input("...")
        pool1KHeight = 1
        pool1KWidth = math.ceil(shape4hConv1[1] / num4EachFM)
        pool1SHeight = 1
        pool1SWidth = math.ceil(shape4hConv1[1] / num4EachFM)
        hROIPooling = self.avg_pool(hConv1,
            pool1KHeight,
            pool1KWidth,
            pool1SHeight,
            pool1SWidth)
        hROIPooling4FCInput = tf.reshape(hROIPooling,
            [self.FLAGS.batchSize,
             -1],  # 这一步的reshape有待验证是否正确
            name = "hROIPooling4FCInput")
        shape4hROIPooling4FCInput = hROIPooling4FCInput.get_shape().as_list()

      # First fully connected layer
      name4VariableScope = "fc1Layer"
      with tf.variable_scope(name4VariableScope):
        name4Weight, name4Bias = "wFC1", "bFC1"
        name4PreAct, name4Act = "preActFC1", "hFC1"

        wFC1 = self.init_weight_variable(name4Weight,
            [shape4ROIPooling4FCInput[1],
             num4FirstFC])
        self.variable_summaries(wFC1)

        bFC1 = self.init_bias_variable(name4Bias,
            [num4FirstFC])
        self.variable_summaries(bFC1)

        preActFC1 = tf.add(tf.matmul(hROIPooling4FCInput, wFC1),
            bFC1,
            name = name4PreAct)
        self.variable_summaries(preActFC1)

        hFC1 = tf.nn.relu(preActFC1,
            name = name4Act)
        self.variable_summaries(hFC1)

      # Second fully connected layer
      name4VariableScope = "fc2Layer"
      with tf.variable_scope(name4VariableScope):
        num4SecondFC = self.FLAGS.num4SecondFC
        name4Weight, name4Bias = "wFC2", "bFC2"
        name4PreAct, name4Act = "preActFC2", "hFC2"

        wFC2 = self.init_weight_variable(name4Weight,
            [num4FirstFC, num4SecondFC])
        self.variable_summaries(wFC2)

        bHidden = self.init_bias_variable(name4Bias,
            [num4SecondFC])
        self.variable_summaries(bFC2)

        preActFC2 = tf.add(tf.matmul(hFC1, wFC2),
            bFC2,
            name = name4PreAct)
        self.variable_summaries(preActFC2)

        hFC2 = tf.nn.relu(preActFC2,
            name = name4Act)
        self.variable_summaries(hFC2)

        hFC2DropOut = tf.nn.dropout(hFC2,
            self.keepProb)
        self.variable_summaries(hFC2DropOut)

      with tf.variable_scope("outputLayer"):
        wOutput = init_weight_variable([hFC2, 2])
        bOutput = init_bias_variable([2])
        hOutput = tf.matmul(hFC2, wOutput) + bOutput
        #hOutput = tf.matmul(hFC2DropOut, wOutput) + bOutput
        yOutput = tf.nn.softmax(hOutput)

      with tf.variable_scope("costLayer"):
        predPro4PandN = tf.reshape(tf.reduce_sum(yOutput, reduction_indices = [0]), [-1, 2])
        predPro4P = tf.matmul(predPro4PandN, tf.constant([[0.], [1.]]))
        predPro4N = tf.matmul(predPro4PandN, tf.constant([[1.], [0.]]))
        predPro4PandNwithLabel = tf.reshape(tf.reduce_sum(self.yLabel * yOutput, reduction_indices = [0]), [-1, 2])
        predPro4PwithLabel = tf.matmul(predPro4PandNwithLabel, tf.constant([[0.], [1.]]))
        predPro4NwithLabel = tf.matmul(predPro4PandNwithLabel, tf.constant([[1.], [0.]]))
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = hOutput, labels = self.yLabel)) - nWeight * predPro4NwithLabel
        trainStep = tf.train.AdamOptimizer(learningRate).minimize(cost)  # GradientDescentOptimizer AdadeltaOptimizer AdamOptimizer
        correctPrediction = tf.equal(tf.argmax(yOutput, 1), tf.argmax(self.yLabel, 1))
        accuracy = tf.reduce_mean(tf.cast(correctPrediction, tf.float32))


