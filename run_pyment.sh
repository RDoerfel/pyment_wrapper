#!/bin/bash

# set directories
FS_subject_dir=/path/to/dir/with/all/freesurfer/recons  # each scan has a folder in here
MR_data_dir=/path/to/dir/with/t1/all/images/    # this folder contains directories for each scan 
script_file=/path/to/pyment_prediction.py   # path to the python script executing
result_dir=/path/to/store/the/results

# set variables
age=20 # not that important
gender=Male # necessary to switch between sex-specific model
image_id=XXX # ID of the scan, should be identical with FS and T1 folder names

# preprocess (bring it into right format)
python $script_file preproc \
    -fs $FS_subject_dir \
    -r $result_dir \
    -id $image_id

# predict (actural age prediction)
python $script_file predict \
    -r $result_dir \
    -id $image_id \
    -a $age \
    -s $gender

