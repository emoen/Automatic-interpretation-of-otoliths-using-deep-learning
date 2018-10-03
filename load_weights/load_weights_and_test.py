from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from keras.preprocessing.image import img_to_array, load_img, array_to_img
from PIL import Image
import scipy

from keras.models import Sequential
from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D, Activation
from keras.models import Model
from keras import optimizers, layers

# Steps to reproduce results from weights
# 1) Normalize images - *(1./255)
# 2) Dont load ImageNet weights
# 3) It is not necessary to set image size on model

x, y = read_img_and_label()
[i for i in range(0,len(x)) if '2006_02135sep2_t05a' in x[i]]
initial_shape=(400, 400, 3)
x_otolith_rescaled = shrink_img_rgb(initial_shape, x[108:110])
x_otolith_rescaled_normalized = np.multiply(x_otolith_rescaled, 1./255)

inception = InceptionV3(include_top=False) #, weights=None) #, input_shape=(512, 512, 3))
z = inception.output
z = GlobalAveragePooling2D()(z)
z = Dense(1)(z)
z = Activation('linear')(z)
otolitt = Model(inputs=inception.input, outputs=z)

adam = optimizers.Adam(lr=0.004, decay= 0.0) #these parameters are irrelevant
otolitt.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy', 'mse', 'mape'])

otolitt.load_weights('./output_checkpoints_inceptionV3_img_400_not_cropped_avgpool_seed8/kfold_-1_400_400_mean_prediction_seed8.006.hdf5')

otolitt.predict(x_otolith_rescaled_normalized)
otolitt.predict(x_otolith_rescaled*1./255)

#otolitt.load_weights('../oto_new_jpeg/output_checkpoints_reproduce_line7_400_400/400_400_mean_prediction.012.hdf5')
#3 [[ 3.45442367]] [[ 3.53675961]] [[ 3.49559164]] 2006_02135sep2_t05a 2006_02135sep2_t05b

def read_img_and_label():

    home_path = "."
    split_df = pd.read_csv(os.path.join(home_path, "split_batch1.csv"))
    split_batch2_df = pd.read_csv(os.path.join(home_path, "split_batch2.csv") )
    split_batch3_df = pd.read_csv(os.path.join(home_path, "split_batch3.csv") )
    split_batch4_df = pd.read_csv(os.path.join(home_path, "split_batch4.csv") )
    
    x1 = split_df['img'].values
    x1 = x1.tolist()
    x1 = [os.path.join(home_path, "batch1", i) for i in x1 ] 
    y1 = split_df['age'].values
    
    x2 = split_batch2_df['img'].values
    x2 = x2.tolist()
    x2 = [os.path.join(home_path, "batch2", i) for i in x2] 
    y2 = split_batch2_df['age'].values
    
    x3 = split_batch3_df['img'].values
    x3 = x3.tolist()
    x3 = [os.path.join(home_path, "batch3", i) for i in x3]
    y3 = split_batch3_df['age'].values
    
    x4 = split_batch4_df['img'].values
    x4 = x4.tolist()
    x4 = [os.path.join(home_path, "batch4", i) for i in x4]
    y4 = split_batch4_df['age'].values
    
    x = x1 + x2 + x3 + x4
    y = np.concatenate((y1,y2,y3,y4))
    return x, y
    
def shrink_img_rgb( new_shape, x):
    """reads images as grayscale given by path in x and resizes them to new_shape returning array of numpy arrays of images
    
    >>> initial_shape = (400, 400, 3)
    >>> x = ['./batch1/2006_02098sep1_t01a.jpg', './batch1/2006_02098sep1_t01b.jpg', './batch1/2006_02098sep1_t02a.jpg']
    >>> shrink_img_rgb( initial_shape, x).shape
    (3, 400, 400, 3)
    """
    
    x_otolith_rescaled = np.empty(shape=(len(x),)+new_shape)
    for ex in range(0, len(x)):
        print("image name:"+str(x[ex]))
        pil_img = load_img(x[ex], grayscale=False)
        an_img = img_to_array(pil_img)
        x_otolith_rescaled[ex] = scipy.misc.imresize( an_img, new_shape)
    return x_otolith_rescaled
    