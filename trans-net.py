from matplotlib import pyplot as pp
from PIL import Image
import tensorflow as tf
import ctypes as ct
import numpy as np
import argparse,os,shutil,sys
try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
    
from autoencoder import *

if __name__== "__main__":
    results=get_parser('autoencoder')
    results.debug_shape=False
    with tf.Session() as sess:
        print('Using tensorflow: %s'%tf.__version__)
        ds=Dataset(results.dataset_path,results.mean_file)
        img=ds.images[0][0]
        #declare nn
        keep=tf.placeholder(tf.float32,name='keepProb')
        image=tf.placeholder(tf.float32,[None,img.shape[0],img.shape[1],img.shape[2]],name='image')
        with tf.variable_scope('encoder',reuse=tf.AUTO_REUSE):
            imresize,feat,h3Shape=encoder(results,image,keep)
        #read model
        load_latest(results,sess)
        im=np.array(Image.open('0001.jpg'))-ds.mean[0]
        print(feat.eval(feed_dict={image:[im],keep:1}))
