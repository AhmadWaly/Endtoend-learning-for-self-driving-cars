from __future__ import division
import numpy as np
# from random import shuffle
from scipy import pi
from scipy.misc import imread, imresize
import os

from sklearn.utils import shuffle
import scipy.io as io

Data_folder='dataset/Hajar'
trainfile=os.path.join(Data_folder,'out.txt')





# Data_folder='driving_dataset'
# trainfile=os.path.join(Data_folder,'data.txt')

batch_pointer=0


X = []
y = []
l=[]
i=0
with open(trainfile) as fp:
    print("textfile opened")
    for line in fp :
        path, angle = line.split()
        x=path.split('.')[0]
        #
        full_path = os.path.join(Data_folder, path)
        X.append(full_path)
        y.append(float(angle) * pi / 180 )
        # X.append('dataset/tmp2/'+line.split(' ')[0])
        # y1=float(line.split(' ')[1])
        # y2=float(line.split(' ')[2])
        # y3=float(line.split(' ')[3])
        # y4=float(line.split(' ')[4])
        # l.append(int(y1))
        # l.append(int(y2))
        # l.append(int(y3))
        # l.append(int(y4))

        # tup = [l[i],l[i+1],l[i+2],l[i+3]]
        # label_index=np.argmax(tup)
        # y.append(np.array(tup))
        i+=4



X,y=shuffle(X,y)
def get_val_data(X,y):
    X_test=X[40000:42560]
    images = np.array([np.float32(imresize(imread(im)[-150:], size=(66, 200))) /255 for im in X_test])
    return np.array(images),np.array(y[40000:42560])
def get_test_data(X,y):
  X_test=X[42560:]
  images = np.array([np.float32(imresize(imread(im)[-150:], size=(66, 200))) /255 for im in X_test])
  return np.array(images),np.array(y[42560:])


def generate_arrays_from_file(path):
    global batch_pointer
    batch_size=64
    train_data=40000

    while True:
        batch_start=batch_pointer%train_data
        batch_end=(batch_pointer+batch_size-1)%train_data
        X_batch=X[batch_start:batch_end]
        images = np.array([np.float32(imresize(imread(im)[-150:], size=(66, 200))) /255 for im in X_batch])

        y_batch = y[batch_start:batch_end]
        batch_pointer+=batch_size
        batch_pointer%=train_data
        if(batch_start%2000==0):
          print('=>',end='')

        yield np.array(images),np.array(y_batch)


print('done')
