# Zoommer Sports complete ML Pipeline 

This directory provides low-code solutions to the following tasks:

1. Cleaning and sorting data
2. Detecting peaks from the cleaned data and extracting segments between two consecutive peaks
3. Extracting features from the segments (including rms, mean, std, skewness, kurtosis, etc.)
4. Compiling these features for each axis of motion and using those to train a K-means anomaly detection model

The end result after running the entire pipeline is a singular .tflite file for each model, that can then be run and deployed in a flutter app using the tflite plugin.

Each exercise has their own README.md for further instructions regarding finetuning of parameters, the effect of this finetuning, and any change in approach or algorithm for that particular exercise.

## Setup 

1. Clone the repository
2. Create a new directory called DATA in the root directory
3. Transfer all data collected using the Data Collection App to this DATA directory. 
4. From the root directory itself, run the cleaning_and_sorting_pipeline.py file. 
5. The data will get sorted into its respective directories (irrespective of data being present in the previous directories or not). 

## Running the pipeline

Follow each model's individual instructions to understand the finetuning of parameters and the effect of this finetuning. In general, each directory has 3 python files -

1. peak detection
2. feature extraction
3. anomaly detection

Running the three files in sequence should get you the tflite file with the anomaly detection model. The directories with plots as part of their name also contain plots with peaks marked and extracted segments - to help understand the peak detection and make changes as required. 

The end file i.e. the Anomaly detection file, will also give an accuracy rate of the model (evaluated only on data for that exercise itself). To test on anomalous data, manual intervention will be required. 

It will also give you a anomaly threshold based on a percentile (calculated from the data set) set in the program. This is adjustable, and is purely to provide a baseline for when deploying and testing the model in the app.