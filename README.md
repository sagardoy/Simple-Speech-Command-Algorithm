
# Udacity Capstone Project - Simple Speech Recognition Algorithm

## Purpose

As independent makers and entrepreneurs develop their products to feature a speech recognition interfacee, many find that their needs for voice commands are minimal – only limited to a handful of basic operational commands. Many robust voice recognition algorithms are extraneous, costly, and unnecessary for many projects. Therefore, an opportunity exists to develop an open-source speech recognition algorithm intended for smaller developers and entry-level enthusiasts.

The purpose of this project was to build an algorithm that can understand a small library of 10 simple commands while being able to ignore, or distinguish from, unknown commands. I submitted this work for the Udacity Machine Learning Engineer Nanodegree in June 2018.

___

## Project Data Source

This project uses the Speech Commands Dataset, a set of 64,721 1-second .wav audio files each consisting of 1-of-30 single spoken English words. These words are from a small set of commands and are spoken by a variety of different speakers. The audio files are organized into folders based on the word they contain, and this data set is designed to help train simple machine learning models. 6 additional audio files for different types of common noise were also included, though not employed in this project. The dataset for this project was downloaded at https://www.kaggle.com/c/tensorflow-speech-recognition-challenge/data. Data was released by Google on August 3, 2017.

___

## File Structure and Contents

The organization and contents of this project folder are as follows:

- root - contains my working project code, algorithm code, project proposal, project report, and .html grab of my project code at end of project
- - .ipynb_checkpoints - contains saved checkpoints for Jupyter Notebook
- - audio - contains lists of partitioned data sets and audio subfolder
- - - audio - folder for downloaded Speech Command Dataset .wav files; files were organized in subfolders by word
- - data - contains decoded audio data of each paritioned dataset formatted as a collection of (79,12) Numpy arrays of mel frequency cepstral coefficient; also contains targets formatted as (1,11) one-hot encoded arrays
- - models - contains 81 trained models in .hdf5 format; models consist of combination of 3 model architectures, 3 different dropout rates, 3 different learning rates, and 3 different leaky ReLU alphas; the weights of each model minimized validation data log loss either naturally or due to early stopping
- - tyler_audio - contains 30 audio clips of me saying each of the 30 words in the Speech Commands Dataset once; these files were used by me to demonstrate and review the final algorithm
            
___

## Dependencies and Setup

To run any of the code found in this project, the following libaries must be downloaded in your environment:

- SciPy (v0.19.1) - Python's scientific computing library; used in this project to directly decode PCM-encoded audio files at a rate of 16000 Mbps and to perform Discrete Cosine Transforms during MFCC calculations
- Numpy (v1.13.3) - library useful for handling and processing arrays of values and objects; used in this project to manage data and for performing Discrete Fourier Transforms during MFCC calculations
- Keras (v2.1.6) with TensorFlow (v1.8.0) backend - Keras provides a high-level programming interface for deep learning and neural network development; Keras runs on top of TensorFlow, a library and engine for dataflow programming; used in this project to develop and train the neural networks and for one-hot encoding all target labels
- Glob - standard Python file searching library; used in this project to extrapolate filenames and paths for all audio files
- OS - standard Python operating system interface; used in this project to combine strings and values into valid file pathnames
- Math - standard Python mathematical functions; used in this project to calculate frequency bins in MFCC calculations

In this project, Keras requires Tensorflow as the backend. To set TensorFlow as the backend, define the environment variable KERAS_BACKEND in your iPython kernel:

``` 
KERAS_BACKEND=tensorflow python -c "from keras import backend" 
```

Additional documentation on Keras's backend may be found at https://keras.io/backend/. 

___

## Usage

Once setup and data downloaded and organized in appropriate folders, use the Jupyter Notebook and sequentially run the cells of either the project code or the algorithm code.
