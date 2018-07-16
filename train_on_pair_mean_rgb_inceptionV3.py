from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import KFold

from keras.models import Sequential
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from keras.applications.imagenet_utils import decode_predictions
from keras import backend as K

from logoutput import log_training
from grid_search_hyperparameters import grid_search_inceptionV3
from InceptionV3_for_regression import InceptionV3_for_regression
from Densenet_for_regression import Densenet_for_regression
from InceptionV4_for_regression import InceptionV4_for_regression
from shrink_img import shrink_img_rgb
from find_pairs_of_otoliths import find_pairs_of_otoliths
from train_val_test_split import train_validate_test_split
from linearize_pairs_to_singles import dataframe_to_nparray_for_pairs_to_singles, dataframe_to_nparray

log_dir = "log_validate16_inceptionV3_avgpooling_nozca_height_shift_100_img_299"
checkpoint_dir = "output_checkpoints_validate16_inceptionV3_avgpooling_nozca_height_shift_100_img_299"
tensorboard_dir = "output_tensorboard_validate16_inceptionV3_avgpooling_zca_nozca_height_shift_100_img_299"
checkpoint_filename = "487_729_mean_prediction"
h5_filename = "./"+log_dir+"/4000_steps_150_epochs_split_rgb_InceptionV3_299_299_no_zca_no_pairs_01.h5"
    
initial_shape=(299, 299, 3) 
def train( do_grid_search=False):
    global initial_shape, log_dir, checkpoint_dir

    x, y = read_img_and_label()
    print("len x:"+str(len(x)))
    print("len y:"+str(len(y)))

    #x = x[0:2000]
    #y = y[0:2000]

    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    a_batch_size = 20

    x_otolith_rescaled = shrink_img_rgb(initial_shape, x)

    pairs, unpaired = find_pairs_of_otoliths(x_otolith_rescaled, x,  y)
    print("len(pairs)"+str(len(pairs)))
    assert (len(pairs)*2) + len(unpaired) == len(x)
    
    df_train_x, df_validation_x, df_test_x = train_validate_test_split(pairs)

    validation_single_x, validation_single_y, idx_validation_single = dataframe_to_nparray_for_pairs_to_singles(df_validation_x)
    assert len(validation_single_x) == len(df_validation_x) * 2
    #TODO: fix
    dev_x = validation_single_x
    validation_y = validation_single_y
    idx_dev = idx_validation_single
    assert len(validation_single_x)  == len(df_validation_x) *2
    
    #train_x has shape (num_train, width, height, 3)
    train_x, train_y, _ = dataframe_to_nparray_for_pairs_to_singles(df_train_x)
    assert len(train_x)  == len(df_train_x) *2
    singles_x, singles_y, _ = dataframe_to_nparray(unpaired)
    assert len(singles_x) == len(unpaired) 
    train_x = np.vstack((train_x, singles_x))
    #train_y.shape = (7560,) singles_y.shape = (657,)
    train_y = np.vstack((train_y[:,None], singles_y[:,None]))
    #train_y.shape == (8217,1)
    assert len(train_x) == len(df_train_x) *2 + len(singles_x)
    assert train_x.shape[0] == len(df_train_x) *2 + len(singles_x)
    assert train_x.shape[0] + validation_single_x.shape[0] + (df_test_x.shape[0]*2) == len(x)
    
    if do_grid_search:
        grid_search_inceptionV3( InceptionV3_for_regression, x_otolith_rescaled[0:2000], y[0:2000])
        return
    
    print("len train_x:"+str(len(train_x)))
    print("len dev_x:"+str(len(dev_x)))

    otolitt, history_callback  = evaluate_as_single_step(train_x, train_y, dev_x, validation_y, a_batch_size, idx_dev, x, y)
    #evaluate_as_kfold(train_datagen, train_x, train_y, dev_x, validation_y, a_batch_size)

    dev_mean_mse= "./"+log_dir+"/dev_mean_MSE_inceptionV3.txt"
    dev_mean_file = open(dev_mean_mse, "w")
    predict_otolitts(df_validation_x, dev_mean_file, otolitt)

    val_mse_history = history_callback.history['val_mean_squared_error']
    numpy_val_mse_history = np.array(val_mse_history)

    min_mse_arg = np.argmin(numpy_val_mse_history)

    dev_mean_file.write("min MSE "+str(np.amin(numpy_val_mse_history)) +" on dev set\n")
    dev_mean_file.write("arg min MSE "+str(min_mse_arg) +" on dev set\n")
    dev_mean_file.write("predict with best fit from dev set:...\n")
    minH5 = './'+checkpoint_dir+'/'+checkpoint_filename+'.'+str(min_mse_arg+1).zfill(3)+'.hdf5'

    otolitt.load_weights(minH5)
    dev_best_mean_file = open("./"+log_dir+"/dev_best_mean_MSE_incetionV3.txt", "w")
    dev_best_mean_file.write("min MSE "+str(np.amin(numpy_val_mse_history)) +" on dev set\n")
    dev_best_mean_file.write("arg min MSE "+str(min_mse_arg) +" on dev set\n")
    dev_best_mean_file.write("predict with best fit from dev set:\n")

    predict_otolitts(df_validation_x, dev_best_mean_file, otolitt)
    dev_mean_file.close()
    dev_best_mean_file.close()

    test_mean_file = open('./'+log_dir+'/test_validate_best_MSE.txt', "w")
    test_mean_file.write("min MSE "+str(np.amin(numpy_val_mse_history)) +" on dev set\n")
    test_mean_file.write("arg min MSE "+str(min_mse_arg) +" on dev set\n")
    test_mean_file.write("predict with best fit from **TEST** set:\n")

    test_mean_file.write("mean train:"+str(sum(train_y)/len(train_y))+"\n")
    test_mean_file.write("mean dev:"+str(sum(validation_y)/len(validation_y))+" length+"+str(len(validation_y))+"\n")
    test_mean_file.write("mean test:"+str(sum(test_y)/len(test_y))+" length:"+str(len(test_y))+"\n")

    predict_otolitts(df_test_x, test_mean_file, otolitt)
    test_mean_file.close()

def predict_otolitts(dev_or_test_set, logfile, otolitt):
    logfile.write("y pred_a pred_b ped_mean \n")

    mse = 0.
    mean_age = 0.
    all_preds = []
    for anIndex, row in dev_or_test_set.iterrows():
        x_otolith_rescaled_a = row['image_vector_a']
        x_otolith_rescaled_b = row['image_vector_b']

        x_otolith_rescaled_a = x_otolith_rescaled_a * 1./255
        x_otolith_rescaled_b = x_otolith_rescaled_b * 1./255

        extra_dim_x_otolith_rescaled_a = np.expand_dims(x_otolith_rescaled_a, axis=0)
        extra_dim_x_otolith_rescaled_b = np.expand_dims(x_otolith_rescaled_b, axis=0)
        pred_a = otolitt.predict( extra_dim_x_otolith_rescaled_a )
        pred_b = otolitt.predict( extra_dim_x_otolith_rescaled_b )
        pred_mean = (pred_a + pred_b)  / 2.0
        all_preds.append(pred_a)
        all_preds.append(pred_b)

        mse += (row['y'] - pred_mean ) **2
        mean_age += row['y']

        logfile.write(str(row['y'])+" "+str(pred_a)+" "+str(pred_b)+" "+str(pred_mean)+" "+row['filename_a']+" "+row['filename_b']+"\n")

    mse = mse * 1./len(dev_or_test_set)
    mean_age = mean_age * 1./len(dev_or_test_set)

    logfile.write("Size "+str(len(dev_or_test_set))+ " of dev_test set:\n")
    logfile.write("after mean with pairs - MSE:"+ (str( mse ))+"\n")
    logfile.write("mean age:" + str(mean_age)+"\n")
    logfile.write("Cov:"+str( np.sqrt(abs(mse)) / mean_age)+"\n")

def evaluate_as_single_step(train_x, train_y, dev_x, validation_y, a_batch_size, idx_dev, x, y):
    global initial_shape, tensorboard_dir, checkpoint_dir

    # Helper: Stop when we stop learning.
    early_stopper = EarlyStopping(patience=20)
    tensorboard = TensorBoard(log_dir='./'+tensorboard_dir+'/487_729_tensorboard_mean_prediction.log')
    checkpointer = ModelCheckpoint(
        filepath = './'+checkpoint_dir+'/'+checkpoint_filename+'.{epoch:03d}.hdf5',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            period=1)

    train_datagen = ImageDataGenerator(
        zca_whitening=False,
        width_shift_range=0.,
        height_shift_range=100,
        zoom_range=0.,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255)
    train_generator = train_datagen.flow(train_x, train_y, batch_size= a_batch_size)

    test_datagen = ImageDataGenerator(
        zca_whitening=False,
        width_shift_range=0.,
        height_shift_range=0.,
        zoom_range=0.,
        rotation_range=360,
        horizontal_flip=True,
        vertical_flip=True,
        rescale=1./255)
    #train_datagen.fit(train_x)

    test_generator = test_datagen.flow(dev_x, validation_y, batch_size= a_batch_size)

    otolitt = InceptionV3_for_regression(the_input_shape=initial_shape)

    history_callback = otolitt.fit_generator(train_generator,
            steps_per_epoch=1600,
            epochs=150,
            callbacks=[early_stopper, tensorboard,checkpointer],
            validation_data=test_generator,
            validation_steps=len(test_generator))

    otolitt.save_weights(h5_filename)
    test_metrics = otolitt.evaluate_generator( test_generator )
    print("len test_generator:"+str(len(test_generator)))
    print("list of metrics:"+str(otolitt.metrics_names))
    print("test metrics:"+str(test_metrics))
    print("history keys:"+str(history_callback.history.keys()))
    #print("history:"+str(history_callback.history))

    log_training(history_callback, otolitt, dev_x, validation_y, test_metrics, -1, idx_dev, x, y, False)
    #print(str(otolitt.predict(dev_x[0:10]*1./255)))
    print("finnished eval")

    return otolitt, history_callback

def read_img_and_label():
    home_path = "."
    split_df = pd.read_csv(os.path.join(home_path, "split_batch1.csv"))
    split_batch2_df = pd.read_csv(os.path.join(home_path, "split_batch2.csv") )
    split_batch3_df = pd.read_csv(os.path.join(home_path, "split_batch3.csv") )
    split_batch4_df = pd.read_csv(os.path.join(home_path, "split_batch4.csv") )
    
    x1 = split_df['img'].values
    x1 = x1.tolist()
    x1 = [os.path.join(home_path, "split_and_crop1", i) for i in x1 ] 
    y1 = split_df['age'].values
    
    x2 = split_batch2_df['img'].values
    x2 = x2.tolist()
    x2 = [os.path.join(home_path, "split_and_crop2", i) for i in x2] 
    y2 = split_batch2_df['age'].values
    
    x3 = split_batch3_df['img'].values
    x3 = x3.tolist()
    x3 = [os.path.join(home_path, "split_and_crop3", i) for i in x3]
    y3 = split_batch3_df['age'].values
    
    x4 = split_batch4_df['img'].values
    x4 = x4.tolist()
    x4 = [os.path.join(home_path, "split_and_crop4", i) for i in x4]
    y4 = split_batch4_df['age'].values
    
    x = x1 + x2 + x3 + x4
    y = np.concatenate((y1,y2,y3,y4))
    return x, y

if __name__ == '__main__':
    train()
    print("hi")

