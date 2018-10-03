from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import GridSearchCV
import copy
from keras.applications.inception_v3 import InceptionV3
from keras.models import Sequential
from keras.layers import Dense

from sklearn.metrics import mean_squared_error, make_scorer

def grid_search_inceptionV3( amodel_fn, X, Y ):
    
    sk_model = KerasRegressor(build_fn=amodel_fn, epochs=30,  batch_size=8) 

    epochs = [30]
    batch_size = [8]
    #optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    #optimizer = ['SGD', 'RMSprop', 'Adam']
    #param_grid_optimizer = dict(optimizer=optimizer)

    param_grid = dict(batch_size=batch_size, epochs=epochs)
    
    #learn_rate = [0.001, 0.01, 0.1, 0.2, 0.3]
    learn_rate = [0.001, 0.0004, 0.0001]
    momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9] 
    de = [0.0, 0.0001, 0.00004, 0.00001]
    param_grid2 = dict(decay=de)  #, momentum=momentum)

    param_grid_4params = dict(epochs=epochs, batch_size=batch_size, learning_rate=learn_rate, decay=de )

    param_grid = dict(learning_rate=learn_rate)   

    grid = GridSearchCV(estimator=sk_model, param_grid=param_grid, scoring=make_scorer(mean_squared_error) , n_jobs=1)
    print("len x:"+str(len(X)))
    print("y:"+str(Y))
    
    grid_result = grid.fit(X, Y)
    

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

