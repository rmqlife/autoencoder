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
from util import *


def stack_vector(mat,vec):
    mat = np.vstack((mat,vec)) if mat.size else vec
    return mat
    
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
        
        
        filelist=get_filelist("dataset/1/")
        filelist = sorted(filelist,key=lambda x: int(os.path.basename(x)[:-4]))
        
        
        tt_feat = np.array([])
        print(filelist)
        for i in range(len(filelist)):
            fn = filelist[i]
            im=np.array(Image.open(fn))-ds.mean[0]
            print("filename", fn)
            feat_val = feat.eval(feed_dict={image:[im],keep:1})
            tt_feat = stack_vector(tt_feat, feat_val)
            print(tt_feat.shape)
        np.save("data", tt_feat)
        
