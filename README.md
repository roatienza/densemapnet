# DenseMapNet
Keras code for my paper on **_"Fast Disparity Estimation using Dense Networks"_** to be presented at the International Conference on Robotics and Automation, Australia, 2018 (ICRA 2018)

**_DenseMapNet_** is a tiny network (only 290k parameters) that can predict disparity (hence, depth) in real-time speed (>30Hz on NVIDIA 1080Ti) given a stereo images.



## Dataset
Please download the dataset from [here](https://drive.google.com/file/d/1zifkJ0duFQAmfZhrr_sOkxOE6qxOc1sT/view?usp=sharing)

Copy on the directory of `densemapnet`.

Extract `tar jxvf dataset.tar.bz2`

At the moment, only Driving dataset is available. Additional datasets will be available in the future.

## Training
Unlike  the test data, training data is split into 4 files. To train the network execute:

`python3 predictor.py --dataset=driving --num_dataset=4`

Alterntaively, load the pre-trained weigths:

`python3 predictor.py --dataset=driving --num_dataset=4 --weights=checkpoint/driving.densemapnet.weights.h5`
