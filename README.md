# DenseMapNet
Keras code for my paper on **_"Fast Disparity Estimation using Dense Networks"_** to be presented at the International Conference on Robotics and Automation, Australia, 2018 (ICRA 2018)

**_DenseMapNet_** is a tiny network (only 293k parameters) that can predict disparity (hence, depth) in real-time speed (>30Hz on NVIDIA 1080Ti) given stereo images of resolution of 960 x 540 RGB.

Sample predictions on different datasets: 
![Sample predictions](https://github.com/roatienza/densemapnet/blob/master/media/Driving.png)

# Demo

<a href="http://www.youtube.com/watch?feature=player_embedded&v=kb7zZ2VqRBg
" target="_blank"><img src="http://img.youtube.com/vi/kb7zZ2VqRBg/0.jpg" 
alt="DenseMapNet Demo" width="640" height="360" border="10" /></a>

## Dataset
Download datasets:
1. [`driving`](https://drive.google.com/file/d/1q01ffNwvnZkrdw58_LIX-tf-vkzsGGmI/view?usp=sharing)
2. [`mpi`](https://drive.google.com/file/d/1mntUmDxpmCPafYh9nCDWPgT6JyzVovDK/view?usp=sharing)

Copy: `cp driving.tar.bz2 densemapnet/dataset`

Change dir and extract: `cd densemanpnet/dataset; tar jxvf driving.tar.bz2`

Available datasets:

1. `driving` - [Driving](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html) 
2. `mpi` - [MPI Sintel](http://sintel.is.tue.mpg.de/)

Additional datasets will be available in the future.

## Training
In some datasets, the train data is split into multiple files. For example, `driving` is split into 4 files while `mpi` fits into 1 file.

To train the network:

`python3 predictor.py --dataset=driving --num_dataset=4`

Alterntaively, load the pre-trained weigths:

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5`

## Testing

To measure EPE using test set:

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5 --notrain`

To benchmark speed only:

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5 --predict`

To generate disparity predictions on both train and test datasets (complete sequential images used to create the video):

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5 --predict
--images`
