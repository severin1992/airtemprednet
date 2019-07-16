'''
Code for processing staged ERA5 data 
'''

import os
import numpy as np
import hickle as hkl
from netCDF4 import Dataset
from settings import *


#Path of ERA5 Data
DATA_DIR = '' #adapt path for your own path!
print(DATA_DIR)
#Path where to save processed data
filingPath = '' #adapt path for your own path!

#create a list of all images
imageList = list(os.walk(DATA_DIR, topdown=False))[-1][-1]
imageList = sorted(imageList)
print('Image List:')
print(imageList)
print('Length of Image List: ' + str(len(imageList)))


#Train,Val,Test size in percentage
partition = [0.8, 0.05, 0.15]
#determine correct indices 
train_begin = 0
train_end = round(partition[0]*len(imageList))-1
val_begin = train_end + 1
val_end = train_end + round(partition[1]*len(imageList))
test_begin = val_end + 1
test_end = len(imageList)-1
print('Indices of Train, Val and test: '+ str(train_begin) + ' ' + str(val_begin) + ' ' + str(test_begin))
#slightly adapting start and end because starts at the first index given and stops before(!) the last. 
train_recordings = imageList[train_begin:val_begin]
val_recordings = imageList[val_begin:test_begin]
test_recordings = imageList[test_begin:test_end]

#adapted for feature testing: just first year (2015); Otherwise would take too long and some weird mistake in some data in 2016
#in total: 17544
#half: 8772
#train: 0-6900
#val:6901-7000
#test:7001-8772
#train_recordings = imageList[0:1000]
#val_recordings = imageList[6901:7000]
#test_recordings = imageList[7001:8772]

print('Now everything together:')
print('Train:')
print(train_recordings)
print('Val:')
print(val_recordings)
print('Test:')
print(test_recordings)

desired_im_sz = (128, 160)
# Create image datasets.
# Processes images and saves them in train, val, test splits.
def process_data():
    splits = {s: [] for s in ['train', 'test', 'val']}
    splits['val'] = val_recordings
    splits['test'] = test_recordings
    splits['train'] = train_recordings
    for split in splits:
        source_list = [DATA_DIR] * len(splits[split]) # corresponds to recording that image came from
        print(splits[split])
        print(source_list)
        print((len(splits[split])==(len(source_list))))
        print('The list of ' + split + ' has length: ' + str(len(source_list)))
        print( 'Creating ' + split + ' data: ' + str(len(source_list)) + ' images')

        # iterate over split and read every .nc file, cut out array, 
        # overlay arrays for RGB like style. 
        # Save everything after for loop.
        EU_stack_list = [0] * (len(splits[split]))

        for i, im_file in enumerate(splits[split]):
            im_path = os.path.join(DATA_DIR, im_file)
            print('Open following dataset: ' + im_path)
            im = Dataset(im_path, mode = 'r')
            #print(im)
            t2 = im.variables['T2'][0,:,:]
            msl = im.variables['MSL'][0,:,:]
            gph500 = im.variables['gph500'][0,:,:]
            im.close()
            EU_t2 = t2[74:202, 550:710]
            EU_msl = msl[74:202, 550:710]
            EU_gph500 = gph500[74:202, 550:710]
            print(EU_t2.shape, EU_msl.shape, EU_gph500.shape)

            ### Outcomment the code that is required in accordance to the desired stack ###

            ###Normal stack: T2, MSL & GPH500
            EU_stack = np.stack([EU_t2, EU_msl, EU_gph500],axis=2)
            EU_stack_list[i]=EU_stack
            ###Stack T2 only:
            #EU_stack = np.stack([EU_t2, EU_t2, EU_t2],axis=2)
            #EU_stack_list[i]=EU_stack
            ###Stack T2*2 MSL*1:
            #EU_stack = np.stack([EU_t2, EU_t2, EU_msl],axis=2)
            #EU_stack_list[i]=EU_stack
            #EU_stack = np.stack([EU_t2, EU_msl, EU_msl],axis=2)
            #EU_stack_list[i]=EU_stack
            ###Stack T2*2 gph500*1:
            #EU_stack = np.stack([EU_t2, EU_t2, EU_gph500],axis=2)
            #EU_stack_list[i]=EU_stack
            ###Stack T2*1 gph500*2
            #EU_stack = np.stack([EU_t2, EU_gph500, EU_gph500],axis=2)
            #EU_stack_list[i]=EU_stack
            #print(EU_stack.shape)
            #X[i]=EU_stack #this should be unnecessary
            ###t2_1 stack. Stack t2 with two empty arrays
            #empty_image = np.zeros(shape = (128, 160))
            #EU_stack = np.stack([EU_t2, empty_image, empty_image],axis=2)
            #EU_stack_list[i]=EU_stack
            ###t2_2 stack. Stack t2 with one empty array
            #empty_image = np.zeros(shape = (128, 160))
            #EU_stack = np.stack([EU_t2, EU_t2, empty_image],axis=2)
            #EU_stack_list[i]=EU_stack
            #print('Does ist work? ')
            #print(EU_stack_list[i][:,:,0]==EU_t2)
            #print(EU_stack[:,:,1]==EU_msl)
        X = np.array(EU_stack_list)
        print('Shape of X: ' + str(X.shape))
        hkl.dump(X, os.path.join(filingPath, 'X_' + split + '.hkl')) #Not optimal!
        hkl.dump(source_list, os.path.join(filingPath, 'sources_' + split + '.hkl'))


if __name__ == '__main__':
    #download_data()
    #extract_data()
    process_data()
