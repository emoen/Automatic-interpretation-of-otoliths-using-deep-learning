import numpy as np

what_is_new = "test-validate size 8% 150 epochs, inceptionV3 linear 400x400 RGB training, GlobalAveragePooling, Adam, mean of pairs"
today_date = "21sept"
type_output = "linear"
network = "inceptionV3_"
image_format= "_RGB_"

adir = "./log_inceptionV3_img_400_not_cropped_avgpool_seed8_tmp/"
date_img_network_typeoutput = today_date + image_format + network+type_output
loss_history_filename = adir + "loss_history_"+date_img_network_typeoutput
val_loss_history_filename = adir + "val_loss_history"+date_img_network_typeoutput
mse_filename = adir + "mse_history_"+date_img_network_typeoutput
val_mse_filename = adir + "val_mse_history_"+date_img_network_typeoutput
acc_filename = adir  + "acc_history_"+date_img_network_typeoutput
val_acc_filename = adir + "val_acc_history"+date_img_network_typeoutput
dev_eval_filename = adir + "dev_evaluation_"+date_img_network_typeoutput

def log_training(history_callback, model, dev_x, dev_y, eval_metrics, kfold_step, idx_dev, x, y, isClass):
    h_keys = history_callback.history.keys()
    if "loss" in h_keys:    
        loss_history = history_callback.history["loss"]
        numpy_loss_history = np.array(loss_history)
        np.savetxt(loss_history_filename + str(kfold_step) + ".txt", numpy_loss_history, delimiter=",")

    if "val_loss" in h_keys:    
        val_loss_history = history_callback.history["val_loss"]
        numpy_val_loss_history = np.array(val_loss_history)
        np.savetxt(val_loss_history_filename + str(kfold_step) + ".txt", numpy_val_loss_history, delimiter=",")
        
    if "mean_squared_error" in h_keys:
        mse_history = history_callback.history['mean_squared_error'] 
        numpy_mse_history = np.array(mse_history)
        np.savetxt(mse_filename + str(kfold_step) + ".txt", numpy_mse_history, delimiter=",")

    if "val_mean_squared_error" in h_keys:
        val_mse_history = history_callback.history['val_mean_squared_error']
        numpy_val_mse_history = np.array(val_mse_history)
        np.savetxt(val_mse_filename +str(kfold_step)+".txt", numpy_val_mse_history, delimiter=",")
    
    if "acc" in h_keys:
        acc_history = history_callback.history['acc']
        numpy_acc_history = np.array(acc_history)
        np.savetxt(acc_filename+str(kfold_step)+".txt", numpy_acc_history, delimiter=",")

    if "val_acc" in h_keys:
        val_acc_history = history_callback.history['val_acc']
        numpy_val_acc_history = np.array(val_acc_history)
        np.savetxt(val_acc_filename+str(kfold_step)+".txt", numpy_val_acc_history, delimiter=",")        

    # When dev-set defined:
    eval = model.evaluate(x = (dev_x * (1./255)), y = dev_y)
    preds = model.predict(dev_x * (1./255))
    file = open(dev_eval_filename+str(kfold_step)+".txt","w") 
    file.write("Changes:"+what_is_new)
    file.write("\n")
    file.write("Eval-metrics:"+str(model.metrics_names))
    file.write("\n")
    file.write("            :"+str(eval))
    file.write("\n")
    file.write("preds mean:"+str(np.mean(preds))+" std:"+str(np.std(preds)))
    file.write("\n")
    file.write("eval_metrics generator: "+str(model.metrics_names))
    file.write("\n")
    file.write("metic_values generator: "+str(eval_metrics))
    file.write("dev_x predictions:\n")
    
    if isClass == True:
        one_hot_inv = np.arange(len(set(y)))
        one_hot_inv +=1
        for i in range(len(preds)):
            y_tilde = np.dot(one_hot_inv, preds[i])
            file.write("^y:"+str(y_tilde)+" y:"+str(y[idx_dev[i]])+" -filename:"+str(x[idx_dev[i]]) +" \n")
    else: 
        for i in range(len(preds)):
            file.write("^y:"+str(preds[i])+" y:"+str(y[idx_dev[i]])+" -filename:"+str(x[idx_dev[i]]) +"\n")
    file.write("\n")
    file.close() 
    print("finnished logging")
