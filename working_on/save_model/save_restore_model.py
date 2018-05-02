
'''
Save and Restore a model using TensorFlow.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from pathlib import Path
from ImgLoader import ImgLoader_28X28InvGray
import matplotlib.pyplot as plt
from MNIST_ClassifierNetwork import MNIST_CN_GetNetwork,MNIST_CN_Traning
import os
import sys

model_path = "./tmp/model.ckpt"

x,y,pred,cost,optimizer = MNIST_CN_GetNetwork()

if(not os.path.isfile(model_path+".meta")):
    #Cannot find a trained network file, train it with 40 epoch.
    TrainEchoCount=40
    MNIST_CN_Traning(x,y,pred,cost,optimizer,TrainEchoCount,model_path,model_path)


batch_x=[]
for filePath in sys.argv[1:]:
    batch_x.append(ImgLoader_28X28InvGray(filePath))
batch_x=np.array(batch_x)


showImg=False
if showImg == True:
    fig = plt.figure()
    for idx,sampleImg in enumerate(batch_x):
        pixels = sampleImg.reshape((28, 28))
        plt.subplot(batch_x.shape[0],1, idx+1)
        plt.imshow(pixels, cmap='gray')
    plt.show()

    def Print784Arr(arr):
        i = 0
        while i < 784:
            if i %28 == 0 :
                print()
            print('@' if (arr[i]>0)else '_'," ", end='')
            i = i + 1

    Print784Arr(batch_x[0]);




def modelPredicting(model_import_path,dat_src):
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    # 'Saver' op to save and restore all the variables
    saver = tf.train.Saver()
    # Running first session
    print("Starting modelPredicting...")
    print("model_import_path:",model_import_path)
    #print("dat_src:",dat_src)

    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        saver.restore(sess, model_import_path)
        # Test model
        predict_result = pred.eval({x: dat_src})
        predict_idx = np.argmax(predict_result, axis=1)
        print("pred:",predict_result, " idx:", predict_idx)
        return predict_idx



modelPredicting(model_path,batch_x)
