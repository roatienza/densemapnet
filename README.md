# DenseMapNet
Keras code for my paper on **_"Fast Disparity Estimation using Dense Networks"_** to be presented at the International Conference on Robotics and Automation, Australia, 2018 (ICRA 2018)

**_DenseMapNet_** is a tiny network (only 293k parameters) that can predict disparity (hence, depth) in real-time speed (>30Hz on NVIDIA 1080Ti) given stereo images of resolution of 960 x 540 RGB.

Sample predictions on different datasets: 
![alt text](https://github.com/roatienza/densemapnet/blob/master/media/Driving.png "Sample predictions")

## Dataset
Please download the dataset:
1. [`driving`](https://drive.google.com/file/d/1q01ffNwvnZkrdw58_LIX-tf-vkzsGGmI/view?usp=sharing)
2. [`mpi`](https://drive.google.com/file/d/1mntUmDxpmCPafYh9nCDWPgT6JyzVovDK/view?usp=sharing)

Copy on the directory of `densemapnet/dataset`.

Extract: `tar jxvf driving.tar.bz2`

Available datasets:

1. `driving` - [Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) 
2. `mpi` - [MPI Sintel](http://sintel.is.tue.mpg.de/)

Additional datasets will be available in the future.

## Training
Unlike  the test data, training data is split into 4 files. To train the network execute:

`python3 predictor.py --dataset=driving --num_dataset=4`

Alterntaively, load the pre-trained weigths.

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5`

## Testing

To measure EPE using test set:

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5 --notrain`

To benchmark speed only:

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5 --predict`

To generate disparity predictions on both train and test datasets (complete sequential images used to create the video):

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5 --predict
--images`

