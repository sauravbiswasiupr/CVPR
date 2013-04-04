#!/usr/bin/python 
'''The script loads the trained weight matrix after an autoencoder is trained, along with the training and testing data and you can then simply pass in the training and test data to get the hidden representations for the test set '''

import os 
import sys 
import time 
import h5py
import numpy 
import theano 
import theano.tensor as T 
from theano.tensor.shared_randomstreams  import RandomStreams 


def find_hidden_rep(x,weights , bias):
    '''A function that when passed the input image representation ( as a 1D vector ) will return the hidden value representation from the autoencoder weights ''' 
 
    activation = numpy.dot(x,weights)+bias 
    hidden = 1./(1. + numpy.exp(-activation))
    return hidden 


def generate_hdf_file(color):
   '''Function that takes as parameter the type of color channel being used and generates hdf files accordingly  '''
   #f1 = h5py.File("da_hidden_vals_8004_.h5")
   #hidden_vals = f1['hidden'][:]
   #f1.close() 
   f2 = h5py.File("weights_trainedDA_8004_"+color+".h5")
   trained_weights = f2['weights'][:]
   f2.close()
   f3 = h5py.File("bias_trainedDA_8004_"+color+".h5")
   trained_bias = f3['bias'][:]
   f3.close() 
   f4 = h5py.File("../labels_256_"+color.lower()+".h5")
   labels = f4['labels'][:]
   f4.close() 
   f5 = h5py.File("../images_256_"+color.lower()+".h5")
   images = f5['images'][:]
   f5.close()
   test_set_images = images[6000:]
   test_set_labels = labels[6000:]
   print "Everything opened successfully , now proceed for finding hidden values " 
   hidden_rep=[]  
   for image in test_set_images:
      print "Image shape : " , image.shape
      hidden_rep.append(find_hidden_rep(image,trained_weights,trained_bias))
   hidden_test = numpy.array(hidden_rep)
   #note in the h5 file on grayscale images the dictionary had the key training_set_hidden
   f = h5py.File("test_set_hidden_rep_"+color.lower()+".h5")
   f["test_set_hidden"] = hidden_test
   f.close()
   print "My work is done " 

if __name__ == "__main__":
  colors=['Green','Blue']
  for color in colors: 
      generate_hdf_file(color)

  print "Alles Klar !!! "   
 
