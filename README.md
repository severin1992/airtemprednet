# airtemprednet

This repository contains the adapted [PredNet](https://github.com/coxlab/prednet) from [Lotter et al. (2016)](https://arxiv.org/abs/1605.08104) for the application to air temperature forecasting.
This is done in context of the Master's Thesis: Deep Learning for Future Frame Prediction of Weather Maps

The code and the model data is compatible with Keras 2.0 and Python 3.6. 


## how to run the model

### t+1 predictions
The code is used for processing and training on ECMWF ERA5 reanaylis data. You can also download trained weights by running `download_models.sh` from the orignal [PredNet repository](https://github.com/coxlab/prednet). The model download includes the original weights trained for the t+1 prediction on the KITTI dataset (moving vehicles), the fine-tuned weights trained to extrapolate the predictions for multiple timesteps.

1. **Download and process data**

	Access the [ECMWF archive](https://www.ecmwf.int/en/forecasts/accessing-forecasts/order-historical-datasets) in order to download the required reanalysis data. 
	When you gathered enough data in the netCDF format, process it by means of `process_netCDF.py`. This will result in 3 hickle files for training, validation and test. 

	```
	python process_netCDF.py
	```

2. **Train the model**

	In order to train the model on the reanalysis dat, run `train.py`. This will train a PredNet model for t+1 predictions. The existing weights will be overwritten.

	```
	python train.py
	```

3. **Evaluate the model**

	This outputs the mean squarred error (MSE) and the peak signal-to-noise ratio (PSNR) and creates plots to compare the ground truth with the predictions. 

	```
	python evaluate.py
	```

### multi-step predictions

1. **Train the model**

	By recursively treating the predictions as inputs, the model can also extrapolate further into the future. In `extrap_finetune.py` you can set `extrap_start_time` in order to determine the starting point for the extrapolations. With `nt = 15` and `extrap_start_time = 10` the model starts extrapolating from time-step 10 and the last out corresponds to a t+5 prediction. 

	```
	python evaluate.py
	```

2. **Evaluate the model**

	Run `evaluate_multistep.py` to output the mean squarred error (MSE) of the model's extrapolations and create plots to compare the ground truth with the predictions. 

	```
	python evaluate_multistep.py
	```

