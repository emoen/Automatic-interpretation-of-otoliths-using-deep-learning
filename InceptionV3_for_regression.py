from keras.applications.inception_v3 import InceptionV3
from keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D, AveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense,Input,BatchNormalization
from keras.models import Model
from keras import optimizers, layers

def InceptionV3_for_regression(the_input_shape=None, learning_rate=0.0004, decay=0.0):
    inception = InceptionV3(include_top=False, weights='imagenet', input_shape=the_input_shape)

    z = inception.output
    z = GlobalAveragePooling2D()(z)
    #z = Dense(1024)(z)
    #z = Dropout(0.5)(z)
    #z = Activation('relu')(z)
    #z = Dense(1, input_dim=1024)(z)
    z = Dense(1)(z)
    z = Activation('linear')(z)
    otolitt = Model(inputs=inception.input, outputs=z)

    adam = optimizers.Adam(lr=learning_rate, decay= decay)
    otolitt.compile(loss='mean_squared_error', optimizer=adam, metrics=['accuracy', 'mse', 'mape'])

    #add weight decay
    #for layer in my_model.layers:
    #    if hasattr(layer, 'kernel_regularizer'):
    #        layer.kernel_regularizer= regularizers.l2(weight_decay)
    #otolitt.load_weights('../tmp_oto2/log_300_steps_600_600_inception/4000_steps_300_epochs_split_rgb_inceptionV3_600_600_01.h5')
    for layer in otolitt.layers:
        layer.trainable = True

    return otolitt

