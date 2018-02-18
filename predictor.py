'''DenseMapNet - a tiny network for fast disparity estimation
from stereo images

Predictor class manages the data, training and prediction

Atienza, R. "Fast Disparity Estimation using Dense Networks".
International Conference on Robotics and Automation,
Brisbane, Australia, 2018.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import keras
from keras.callbacks import ModelCheckpoint, LambdaCallback
from keras.optimizers import RMSprop, SGD

import numpy as np

import argparse
import os
from os import path
import time
import matplotlib.image as img
import matplotlib.pyplot as plt
from scipy import misc

from utils import Settings
from utils import ElapsedTimer
from densemapnet import DenseMapNet


class Predictor(object):
    def __init__(self, settings=Settings()):
        self.settings = settings
        self.mkdir_images()
        self.pdir = "dataset" 
        self.get_max_disparity()
        self.load_test_data()
        self.network  = None
        self.train_data_loaded = False

    def load_test_disparity(self):
        filename = self.settings.dataset + ".test.disparity.npz"
        print("Loading... ", filename)
        self.test_dx = np.load(os.path.join(self.pdir, filename))['arr_0']
        self.dmax =  max(self.dmax, np.amax(self.test_dx))
        self.dmin =  min(self.dmin, np.amin(self.test_dx))
        self.sum += np.sum(self.test_dx, axis=0)
        self.size += self.test_dx.shape[0]
        print("Max disparity on entire dataset: ", self.dmax)
        print("Min disparity on entire dataset: ", self.dmin)
        if self.settings.predict:
            filename = self.settings.dataset + "_complete.test.disparity.npz"
            print("Loading... ", filename)
            self.test_dx = np.load(os.path.join(self.pdir, filename))['arr_0']

        dim = self.test_dx.shape[1] * self.test_dx.shape[2]
        self.ave = np.sum(self.sum / self.size) / dim
        print("Ave disparity: ", self.ave)
        if self.settings.otanh:
            self.dmax *= 0.5 
            print("Max disparity for tanh: ", self.dmax)
            self.test_dx = self.test_dx.astype('float32') - self.dmax
            self.test_dx = self.test_dx.astype('float32') / self.dmax
            print("Scaled disparity max: ", np.amax(self.test_dx))
            print("Scaled disparity min: ", np.amin(self.test_dx))
            # self.test_dx =  np.clip(self.test_dx, -1.0, 1.0)
        else:
            self.test_dx = self.test_dx.astype('float32') / self.dmax
            print("Scaled disparity max: ", np.amax(self.test_dx))
            print("Scaled disparity min: ", np.amin(self.test_dx))
            # self.test_dx =  np.clip(self.test_dx, 0.0, 1.0)

        shape = [-1, self.test_dx.shape[1], self.test_dx.shape[2], 1]
        self.test_dx = np.reshape(self.test_dx, shape)

    def get_max_disparity(self):
        self.dmax = 0
        self.dmin = 255
        self.sum = None
        self.size = 0
        count = self.settings.num_dataset + 1
        for i in range(1, count, 1):
            filename = self.settings.dataset + ".train.disparity.%d.npz" % i
            print("Loading... ", filename)
            self.train_dx = np.load(os.path.join(self.pdir, filename))['arr_0']
            self.dmax =  max(self.dmax, np.amax(self.train_dx))
            self.dmin =  min(self.dmin, np.amin(self.train_dx))
            if self.sum is None:
                self.sum = np.sum(self.train_dx, axis=0)
            else:
                self.sum += np.sum(self.train_dx, axis=0)
            self.size += self.train_dx.shape[0]
        self.load_test_disparity()

    def load_test_data(self):
        if self.settings.predict:
            filename = self.settings.dataset + "_complete.test.left.npz"
            print("Loading... ", filename)
            self.test_lx = np.load(os.path.join(self.pdir, filename))['arr_0']
            filename = self.settings.dataset + "_complete.test.right.npz"
            print("Loading... ", filename)
            self.test_rx = np.load(os.path.join(self.pdir, filename))['arr_0']
        else:
            filename = self.settings.dataset + ".test.left.npz"
            print("Loading... ", filename)
            self.test_lx = np.load(os.path.join(self.pdir, filename))['arr_0']
            filename = self.settings.dataset + ".test.right.npz"
            print("Loading... ", filename)
            self.test_rx = np.load(os.path.join(self.pdir, filename))['arr_0']
        self.channels = self.settings.channels
        self.xdim = self.settings.xdim = self.test_lx.shape[2]
        self.ydim = self.settings.ydim = self.test_lx.shape[1]

    def load_train_data(self, index):
        filename = self.settings.dataset + ".train.left.%d.npz" % index
        print("Loading... ", filename)
        self.train_lx = np.load(os.path.join(self.pdir, filename))['arr_0']

        filename = self.settings.dataset + ".train.right.%d.npz" % index
        print("Loading... ", filename)
        self.train_rx = np.load(os.path.join(self.pdir, filename))['arr_0']

        filename = self.settings.dataset + ".train.disparity.%d.npz" % index
        print("Loading... ", filename)
        self.train_dx = np.load(os.path.join(self.pdir, filename))['arr_0']

        if self.settings.otanh:
            self.train_dx = self.train_dx.astype('float32') - self.dmax
            self.train_dx = self.train_dx.astype('float32') / self.dmax
            print("Scaled disparity max: ", np.amax(self.train_dx))
            print("Scaled disparity min: ", np.amin(self.train_dx))
            # self.train_dx = np.clip(self.train_dx, -1.0, 1.0)
        else:
            self.train_dx = self.train_dx.astype('float32') / self.dmax
            print("Scaled disparity max: ", np.amax(self.train_dx))
            print("Scaled disparity min: ", np.amin(self.train_dx))
            # self.train_dx = np.clip(self.train_dx, 0.0, 1.0)
        shape =  [-1, self.train_dx.shape[1], self.train_dx.shape[2], 1]
        self.train_dx = np.reshape(self.train_dx, shape)

        self.channels = self.settings.channels = self.train_lx.shape[3]
        self.xdim = self.settings.xdim = self.train_lx.shape[2]
        self.ydim = self.settings.ydim = self.train_lx.shape[1]
        self.train_data_loaded = True

    def train_network(self):
        if self.settings.num_dataset == 1:
            self.train_all()
            return

        # if self.settings.model_weights:
            # if not starting from scratch, better to start at a lower lr
            # lr = 0.5e-4

        epochs = 1
        if self.settings.otanh:
            decay = 0 # 2e-6
            lr = 1e-2 + decay
        else:
            decay = 1e-6
            lr = 1e-3 + decay
        for i in range(400):
            # if decay > 0:
            #    k = i * decay
            #    lr *= (1. / (1. + k))
            lr = lr - decay
            print("Epoch: ", (i+1), " Learning rate: ", lr)
            self.train_batch(epochs=epochs, lr=lr, seq=(i+1))
            self.predict_disparity()

    def train_all(self, epochs=400, lr=1e-3):
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("Folder exists: ", checkdir)

        filename = self.settings.dataset
        filename += ".densemapnet.weights.{epoch:02d}.h5"
        filepath = os.path.join(checkdir, filename)
        checkpoint = ModelCheckpoint(filepath=filepath,
                                     save_weights_only=True,
                                     verbose=1,
                                     save_best_only=False)
        predict_callback = LambdaCallback(on_epoch_end=lambda epoch,
                                          logs: self.predict_disparity())
        callbacks = [checkpoint, predict_callback]
        self.load_train_data(1)
        if self.network is None:
            self.network = DenseMapNet(settings=self.settings)
            self.model = self.network.build_model(lr=lr)

        if self.settings.otanh:
            print("Using loss=mse on tanh output layer")
            self.model.compile(loss='mse',
                               optimizer=RMSprop(lr=lr, decay=1e-6))
        else:
            print("Using loss=crossent on sigmoid output layer")
            self.model.compile(loss='binary_crossentropy',
                               optimizer=RMSprop(lr=lr, decay=1e-6))

        if self.settings.model_weights:
            if self.settings.notrain:
                self.predict_disparity()
                return

        x = [self.train_lx, self.train_rx]
        self.model.fit(x,
                       self.train_dx,
                       epochs=epochs,
                       batch_size=4,
                       shuffle=True,
                       callbacks=callbacks)

    def train_batch(self, epochs=10, lr=1e-3, seq=1):
        count = self.settings.num_dataset + 1
        checkdir = "checkpoint"
        try:
            os.mkdir(checkdir)
        except FileExistsError:
            print("Folder exists: ", checkdir)

        is_model_compiled = False
            
        indexes = np.arange(1,count)
        np.random.shuffle(indexes)
        
        # for i in range(1, count, 1):
        for i in indexes:
            filename = self.settings.dataset
            # filename += ".densemapnet.weights.{epoch:02d}.h5"
            filename += ".densemapnet.weights.%d-%d.h5" % (seq, i)
            filepath = os.path.join(checkdir, filename)
            checkpoint = ModelCheckpoint(filepath=filepath,
                                         save_weights_only=True,
                                         verbose=1,
                                         save_best_only=False)
            callbacks = [checkpoint]

            self.load_train_data(i)

            if self.network is None:
                self.network = DenseMapNet(settings=self.settings)
                self.model = self.network.build_model(lr=lr)

            if not is_model_compiled:
                if self.settings.otanh:
                    sgd = SGD(lr=lr, momentum=0.5, nesterov=True)
                    print("Using loss=mse on tanh output layer")
                    self.model.compile(loss='mse', optimizer=sgd)
                else:
                    print("Using loss=crossent on sigmoid output layer")
                    self.model.compile(loss='binary_crossentropy',
                                       optimizer=RMSprop(lr=lr, decay=1e-6))
                is_model_compiled = True

            if self.settings.model_weights:
                if self.settings.notrain:
                    self.predict_disparity()
                    return

            x = [self.train_lx, self.train_rx]
            self.model.fit(x,
                           self.train_dx,
                           epochs=epochs,
                           batch_size=4,
                           shuffle=True,
                           callbacks=callbacks)

    def mkdir_images(self):
        self.images_pdir = "images"
        pdir = self.images_pdir

        for dirname in ["train", "test"]:
            cdir = os.path.join(pdir, dirname)
            filepath = os.path.join(cdir, "left")
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(cdir, "right")
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(cdir, "disparity")
            os.makedirs(filepath, exist_ok=True)
            filepath = os.path.join(cdir, "prediction")
            os.makedirs(filepath, exist_ok=True)


    def get_epe(self, use_train_data=True, get_performance=False):
        if use_train_data:
            lx = self.train_lx
            rx = self.train_rx
            dx = self.train_dx
            print("Using train data... Size: ", lx.shape[0])
        else:
            lx = self.test_lx
            rx = self.test_rx
            dx = self.test_dx
            if self.settings.predict:
                print("Using complete data... Size: ", lx.shape[0])
            else:
                print("Using test data... Size: ", lx.shape[0])

        # sum of all errors (normalized)
        epe_total = 0
        # count of images
        t = 0
        nsamples = lx.shape[0]
        elapsed_total = 0.0
        if self.settings.images:
            print("Saving images on folder...")
        for i in range(0, nsamples, 1):
            indexes = np.arange(i, i + 1)
            left_images = lx[indexes, :, :, : ]
            right_images = rx[indexes, :, :, : ]
            disparity_images = dx[indexes, :, :, : ]
            # measure the speed of prediction on the 10th sample to avoid variance
            if get_performance:
                start_time = time.time()
                predicted_disparity = self.model.predict([left_images, right_images])
                elapsed_total += (time.time() - start_time)
            else:
                predicted_disparity = self.model.predict([left_images, right_images])

            predicted = predicted_disparity[0, :, :, :]
            if self.settings.dataset == "sparse":
                ground_mask = np.ceil(disparity_images[0, :, :, :])
                predicted = np.multiply(predicted, ground_mask)

            ground = disparity_images[0, :, :, :]
            epe = predicted - ground
            if self.settings.dataset == "sparse":
                dim = np.count_nonzero(ground_mask)
            else:
                dim = predicted.shape[0] * predicted.shape[1]
            # normalized error on all pixels
            epe = np.sum(np.absolute(epe))
            # epe = epe.astype('float32')
            epe = epe / dim
            epe_total += epe

            if (get_performance and self.settings.images) or ((i%100) == 0): 
                path = "test"
                if use_train_data:
                    path = "train"
                filepath  = os.path.join(self.images_pdir, path)
                left = os.path.join(filepath, "left")
                right = os.path.join(filepath, "right")
                disparity = os.path.join(filepath, "disparity")
                prediction = os.path.join(filepath, "prediction")
                filename = "%04d.png" % i
                left = os.path.join(left, filename)
                plt.imsave(left, left_images[0])
                right = os.path.join(right, filename)
                plt.imsave(right, right_images[0])
                self.predict_images(predicted, os.path.join(prediction, filename))
                self.predict_images(ground, os.path.join(disparity, filename))

        epe = epe_total / nsamples 
        # epe in pix units
        epe = epe * self.dmax
        if self.settings.dataset == "sparse":
            epe = epe / 256.0
        print("EPE: %0.2fpix" % epe)
        # speed in sec
        if get_performance:
            print("Speed: %0.4fsec" % (elapsed_total / nsamples))
            print("Speed: %0.4fHz" % (nsamples / elapsed_total))

    def predict_images(self, image, filepath):
        size = [image.shape[0], image.shape[1]]
        if self.settings.otanh:
            image += 1.0
            image =  np.clip(image, 0.0, 2.0)
            image *= (255*0.5)
        else:
            image =  np.clip(image, 0.0, 1.0)
            image *= 255
        image = image.astype(np.uint8)
        image = np.reshape(image, size)
        misc.imsave(filepath, image)

    def predict_disparity(self):
        if self.settings.predict:
            if self.network is None:
                self.network = DenseMapNet(settings=self.settings)
                self.model = self.network.build_model()
            # gpu is slow in prediction during initial load of data
            # distorting the true speed of the network
            # we get the speed after 1 prediction
            if self.settings.images:
                self.get_epe(use_train_data=False, get_performance=True)
            else:
                for i in range(4):
                    self.get_epe(use_train_data=False, get_performance=True)
        else:
            # self.settings.images = True
            self.get_epe(use_train_data=False, get_performance=True)
            # self.get_epe(use_train_data=False)
            if self.settings.notrain:
                self.get_epe()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load checkpoint hdf5 file of model trained weights"
    parser.add_argument("-w",
                        "--weights",
                        help=help_)
    parser.add_argument("-d",
                        "--dataset",
                        help="Name of dataset to load")
    parser.add_argument("-n",
                        "--num_dataset",
                        type=int,
                        help="Number of dataset file splits to load")
    help_ = "No training. Just prediction based on test data. Must load weights."
    parser.add_argument("-p",
                        "--predict",
                        action='store_true',
                        help=help_)
    help_ = "Generate images during prediction. Images are stored images/"
    parser.add_argument("-i",
                        "--images",
                        action='store_true',
                        help=help_)
    help_ = "No training. EPE benchmarking on test set. Must load weights."
    parser.add_argument("-t",
                        "--notrain",
                        action='store_true',
                        help=help_)
    help_ = "Use hyperbolic tan in the output layer"
    parser.add_argument("-o",
                        "--otanh",
                        action='store_true',
                        help=help_)
    help_ = "Sparse ground truth data"
    parser.add_argument("-s",
                        "--sparse",
                        action='store_true',
                        help=help_)
    help_ = "No padding"
    parser.add_argument("-a",
                        "--nopadding",
                        action='store_true',
                        help=help_)
    
    
    args = parser.parse_args()
    settings = Settings()
    settings.model_weights = args.weights
    settings.dataset = args.dataset
    settings.num_dataset = args.num_dataset
    settings.predict = args.predict
    settings.images = args.images
    settings.notrain = args.notrain
    settings.otanh = args.otanh
    settings.sparse = args.sparse
    settings.nopadding = args.nopadding

    predictor = Predictor(settings=settings)
    if settings.predict:
        predictor.predict_disparity()
    else:
        predictor.train_network()
