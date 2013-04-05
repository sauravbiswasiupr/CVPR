README
-------

This repo contains all the data matrices generated from the running of autoencoders on the SICURA image dataset containing 8004 images . The trained weights and biases from the hidden layer for grayscale images and red , green , blue channels is stored here . It also contains the test set hidden reps from the above image dataset for all possible channels as well as grayscale . 

INSTRUCTIONS 
------------
1 . To access any hidden layer rep , please use h5py module to load the .h5 files . So for example for the grayscale color channel ,please use the following lines of code 
>> import h5py 
>> f  = h5py.File("test_set_hidden_rep.h5")
>>hidden_test_grayscale = f["training_set_hidden"][:]


For the red , green and blue color channels please use the following peace of code 

>>import h5py 
>>f = h5py.File("test_set_hidden_rep_red.h5") #replace red with blue or green as reqd
>>test_set_hidden_red = f["test_set_hidden"][:] 


The test_set_hidden variables now contain numpy array of size 2004 * 500 
Each 500 col vector is a feature for a test image passed through the autoencoder and its hidden rep calculated 

For reading the image labels use this 
>>f = h5py.File("labels_256_gray.h5")
>>labels = f["labels"] 

All the labels are same for red , green , blue or grayscale channels but separate named ones have been saved for the sake of brevity 

NOTE : Please use different file pointers to read different .h5 files  

##More updates as more code is done 
