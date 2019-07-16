'''
Evaluate trained PredNet
Calculates mean-squared error and plots predictions.
'''

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from keras import backend as K
from keras.models import Model, model_from_json
from keras.layers import Input, Dense, Flatten

from prednet import PredNet
from data_utils import SequenceGenerator
from settings import *

import math

##Just for checking how the shape is after generator.create_all() from Sequence Generator
import hickle as hkl
##
n_plot = 10 #number of plots
batch_size = 10
nt = 15 #number of timesteps used for sequences in training
extrap = 10 #frame number from where extrapolation will start to be produced



weights_file = os.path.join(WEIGHTS_DIR, 'prednet_netCDF_weights-extrapfinetuned.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_netCDF_model-extrapfinetuned.json')
test_file = os.path.join(DATA_DIR, 'X_test.hkl')
test_sources = os.path.join(DATA_DIR, 'sources_test.hkl')


# Load trained model
f = open(json_file, 'r')
json_string = f.read()
f.close()
train_model = model_from_json(json_string, custom_objects = {'PredNet': PredNet})
train_model.load_weights(weights_file)

# Create testing model (to output predictions)
layer_config = train_model.layers[1].get_config()
print('layer_config: ' + str(layer_config))
layer_config['output_mode'] = 'prediction' #'prediction'
layer_config['extrap_start_time'] = extrap
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
#data_format = 'channels_last' #inserted for experimentation
print(data_format)
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
print('input_shape: ' + str(input_shape))
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
print('inputs: ' + str(inputs)) 
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format) # orig: unique
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
print('Shape X_test: ' + str(X_test.shape))
print('Shape X_hat: ' + str(X_hat.shape))
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))
print('Shape X_test: ' + str(X_test.shape))
print('Shape X_hat: ' + str(X_hat.shape))
##Just for checking how the shape is after generator.create_all() from Sequence Generator
#hkl.dump(X_test, os.path.join(RESULTS_SAVE_DIR, 'X_testMultistep.hkl'))
#hkl.dump(X_hat, os.path.join(RESULTS_SAVE_DIR, 'X_hatMultistep.hkl'))
##

# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
shapeXhat = str(X_hat.shape) #Just have a look at the shapes to be sure we are calculating the right MSE 
shapeXtest = str(X_test.shape) 
mse_model = np.mean( (X_test[:, 1:,:,:,0] - X_hat[:, 1:,:,:,0])**2 )  # look at all timesteps except the first
mse_model1 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 10,:,:,0])**2 )
mse_model2 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 11,:,:,0])**2 )
mse_model3 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 12,:,:,0])**2 )
mse_model4 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 13,:,:,0])**2 )
mse_model5 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 14,:,:,0])**2 )
#mse_model6 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 15,:,:,0])**2 )
#mse_model7 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 16,:,:,0])**2 )
#mse_model8 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 17,:,:,0])**2 )
#mse_model9 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 18,:,:,0])**2 )
#mse_model10 = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 19,:,:,0])**2 )
#mse_model_last = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 19,:,:,0])**2 )
mse_prev1 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 10,:,:,0])**2 )
mse_prev2 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 11,:,:,0])**2 )
mse_prev3 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 12,:,:,0])**2 )
mse_prev4 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 13,:,:,0])**2 )
mse_prev5 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 14,:,:,0])**2 )
#mse_prev6 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 15,:,:,0])**2 )
#mse_prev7 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 16,:,:,0])**2 )
#mse_prev8 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 17,:,:,0])**2 )
#mse_prev9 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 18,:,:,0])**2 )
#mse_prev10 = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 19,:,:,0])**2 )
#mse_prevFromExtrap = np.mean( (X_test[:, 9,:,:,0] - X_test[:, 19,:,:,0])**2 )
mse_prev = np.mean( (X_test[:, :-1,:,:,0] - X_test[:, 1:,:,:,0])**2 )
# Calculate PSNR
# Function to calculate PSNR
# In the absence of noise, the two images I and K are identical, and thus the MSE is zero. In this case the PSNR is infinite. 
# Or here the best value: 100
def psnr(img1, img2):
    mse = np.mean( (img1 - img2) ** 2 )
    if mse == 0: return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
psnr_model = psnr(X_test[:, 1:,:,:,0], X_hat[:, 1:,:,:,0])
psnr_model_last = psnr(X_test[:, 9,:,:,0], X_hat[:, 14,:,:,0])
psnr_prevFromExtrap = psnr(X_test[:, 9,:,:,0], X_test[:, 14,:,:,0])
psnr_prev = psnr(X_test[:, :-1,:,:,0], X_test[:, 1:,:,:,0])

if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(os.path.join(RESULTS_SAVE_DIR, 'prediction_scores.txt'), 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Model MSE t+1: %f\n" % mse_model1)
f.write("Model MSE t+2: %f\n" % mse_model2)
f.write("Model MSE t+3: %f\n" % mse_model3)
f.write("Model MSE t+4: %f\n" % mse_model4)
f.write("Model MSE t+5: %f\n" % mse_model5)
#f.write("Model MSE t+6: %f\n" % mse_model6)
#f.write("Model MSE t+7: %f\n" % mse_model7)
#f.write("Model MSE t+8: %f\n" % mse_model8)
#f.write("Model MSE t+9: %f\n" % mse_model9)
#f.write("Model MSE t+10: %f\n" % mse_model10)
f.write("Prev Frame MSE t+1: %f\n" % mse_prev1)
f.write("Prev Frame MSE t+2: %f\n" % mse_prev2)
f.write("Prev Frame MSE t+3: %f\n" % mse_prev3)
f.write("Prev Frame MSE t+4: %f\n" % mse_prev4)
f.write("Prev Frame MSE t+5: %f\n" % mse_prev5)
#f.write("Prev Frame MSE t+6: %f\n" % mse_prev6)
#f.write("Prev Frame MSE t+7: %f\n" % mse_prev7)
#f.write("Prev Frame MSE t+8: %f\n" % mse_prev8)
#f.write("Prev Frame MSE t+9: %f\n" % mse_prev9)
#f.write("Prev Frame MSE t+10: %f\n" % mse_prev10)
f.write("Previous Frame MSE: %f\n" % mse_prev)
f.write("")
f.write("Model PSNR: %f\n" % psnr_model)
f.write("Model PSNR from only last prediction in sequence: %f\n" % psnr_model_last)
f.write("Previous frame PSNR last frame vs extrap start time: %f\n" % psnr_prevFromExtrap)
f.write("Previous frame PSNR: %f\n" % psnr_prev)
f.write("Shape of X_test: " +  shapeXtest)
f.write("")
f.write("Shape of X_hat: " +  shapeXhat)
f.write("")
f.write("Data format: " + data_format)
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 1,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 2,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 3,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 4,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 5,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 6,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 7,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 8,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 9,:,:,0])**2 )))
f.write("")
f.write(str(np.mean( (X_test[:, 0,:,:,0] - X_test[:, 10,:,:,0])**2 )))
f.close()

# Plot some predictions
aspect_ratio = float(X_hat.shape[2]) / X_hat.shape[3]
plt.figure(figsize = (nt, 2*aspect_ratio))
gs = gridspec.GridSpec(2, nt)
gs.update(wspace=0., hspace=0.)
plot_save_dir = os.path.join(RESULTS_SAVE_DIR, 'prediction_plots/')
if not os.path.exists(plot_save_dir): os.mkdir(plot_save_dir)
plot_idx = np.random.permutation(X_test.shape[0])[:n_plot]
for i in plot_idx:
    for t in range(nt):
        plt.subplot(gs[t])
        plt.imshow(X_test[i,t,:,:,0], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t,:,:,0], interpolation='none')
        plt.tick_params(axis='both', which='both', bottom='off', top='off', left='off', right='off', labelbottom='off', labelleft='off')
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.jpg')
    plt.clf()

