'''
Evaluate trained PredNet on netCDF sequences.
Calculates mean-squared error and PSNR and plots predictions.
'''
import math
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

##Just for checking how the shape is after generator.create_all() from Sequence Generator
import hickle as hkl
##

n_plot = 10
batch_size = 10
nt = 10

weights_file = os.path.join(WEIGHTS_DIR, 'prednet_netCDF_weights.hdf5')
json_file = os.path.join(WEIGHTS_DIR, 'prednet_netCDF_model.json')
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
layer_config['output_mode'] = 'prediction'
data_format = layer_config['data_format'] if 'data_format' in layer_config else layer_config['dim_ordering']
test_prednet = PredNet(weights=train_model.layers[1].get_weights(), **layer_config)
input_shape = list(train_model.layers[0].batch_input_shape[1:])
input_shape[0] = nt
inputs = Input(shape=tuple(input_shape))
predictions = test_prednet(inputs)
test_model = Model(inputs=inputs, outputs=predictions)

test_generator = SequenceGenerator(test_file, test_sources, nt, sequence_start_mode='unique', data_format=data_format)
X_test = test_generator.create_all()
X_hat = test_model.predict(X_test, batch_size)
if data_format == 'channels_first':
    X_test = np.transpose(X_test, (0, 1, 3, 4, 2))
    X_hat = np.transpose(X_hat, (0, 1, 3, 4, 2))

##Just for checking how the shape is after generator.create_all() from Sequence Generator
#hkl.dump(X_test, os.path.join(RESULTS_SAVE_DIR, 'X_AfterGeneratorStandardized.hkl'))
#hkl.dump(X_hat, os.path.join(RESULTS_SAVE_DIR, 'X_hatStandardized.hkl'))
##


# Compare MSE of PredNet predictions vs. using last frame.  Write results to prediction_scores.txt
# Furthermore, calculate Model MSE from the last prediction of the sequence only
# as the model improves after several frames (mse_model_last)
# Typical shape of X_test and X_hat: (263, 10, 128, 160, 3)
# where 263 are the sequences, 10 ist the amount of frames in one sequence, 
# 128 & 160 are the image sice and 3 the number of layers.
# For our case only take layer 0 (= T2) into account. 
shapeXhat = str(X_hat.shape) #Just have a look at the shapes to be sure we are calculating the right MSE 
shapeXtest = str(X_test.shape) 
mse_model = np.mean( (X_test[:, 1:,:,:,0] - X_hat[:, 1:,:,:,0])**2 )  # look at all timesteps except the first
mse_model_last = np.mean( (X_test[:, 9,:,:,0] - X_hat[:, 9,:,:,0])**2 )
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
psnr_model_last = psnr(X_test[:, 9,:,:,0], X_hat[:, 9,:,:,0])
psnr_prev = psnr(X_test[:, :-1,:,:,0], X_test[:, 1:,:,:,0])

if not os.path.exists(RESULTS_SAVE_DIR): os.mkdir(RESULTS_SAVE_DIR)
f = open(RESULTS_SAVE_DIR + 'prediction_scores.txt', 'w')
f.write("Model MSE: %f\n" % mse_model)
f.write("Model MSE from only last prediction in sequence: %f\n" % mse_model_last)
f.write("Previous Frame MSE: %f\n" % mse_prev)
f.write("Model PSNR: %f\n" % psnr_model)
f.write("Model PSNR from only last prediction in sequence: %f\n" % psnr_model_last)
f.write("Previous frame PSNR: %f\n" % psnr_prev)
f.write("Shape of X_test: " +  shapeXtest)
f.write("")
f.write("Shape of X_hat: " +  shapeXhat)
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
        plt.imshow(X_test[i,t,:,:,0], interpolation='none') #the last index sets the channel. 0 = t2
        #plt.pcolormesh(X_test[i,t,::-1,:,0], shading='bottom', cmap=plt.cm.jet)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if t==0: plt.ylabel('Actual', fontsize=10)

        plt.subplot(gs[t + nt])
        plt.imshow(X_hat[i,t,:,:,0], interpolation='none')
        #plt.pcolormesh(X_hat[i,t,::-1,:,0], shading='bottom', cmap=plt.cm.jet)
        plt.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False, labelbottom=False, labelleft=False)
        if t==0: plt.ylabel('Predicted', fontsize=10)

    plt.savefig(plot_save_dir +  'plot_' + str(i) + '.png')
    plt.clf()