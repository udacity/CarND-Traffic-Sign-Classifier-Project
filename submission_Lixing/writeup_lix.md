# **Project 3: Traffic Sign Recognition** 
## Writeup by Ling ZHANG

---
### 1. Project Overview
#### a. Goals
The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

#### b.Structure of the Project
TODO


[//]: # (Image References)

[image1]: ./references/00000_00012.jpg
[image2]: ./references/00002_00025.jpeg
[image3]: ./references/gray.PNG
[image_lenet]:  ./references/LeNet.png
[image_train4]: ./references/results/fourth.jpg
[image_train5]: ./references/results/fifth.jpg
[image_test1]: ./references/test_image/img_1.jpg
[image_test2]: ./references/test_image/img_2.jpg
[image_test3]: ./references/test_image/img_3.jpg
[image_test4]: ./references/test_image/img_4.jpg
[image_test5]: ./references/test_image/img_5.jpg
[image_test1_gray]: references/test_image/normalized/img_1_gray.jpg
[image_test2_gray]: references/test_image/normalized/img_2_gray.jpg
[image_test3_gray]: references/test_image/normalized/img_3_gray.jpg
[image_test4_gray]: references/test_image/normalized/img_4_gray.jpg
[image_test5_gray]: references/test_image/normalized/img_5_gray.jpg

---
### 2. Build-up Explanation

#### 1) Data Preprocessing
##### a. Fetching the data

The Dataset comes from the so-called [GTSRB](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) project. On the official website, there are 5 types of data available both for training set and test set:
 - Images and annotations
 - Three sets of different HOG features
 - Haar-like features
 - Hue histograms
 
 Among which, the *Images and annotation* type is what we want. Both the training set and test set are made up of 2 parts:
 1) The image data with the .ppm ffix. The sizes vary for each picture. Here is an example from class 5:
 
    ![][image1] 
 2) The anntations inside the .csv file, which consists of the following information:
 
| Filename  | Width     | Height    | Roi.X1    | Roi.Y1    | Roi.X2    | Roi.Y2    | ClassId   |
|:----------|:----------|:----------|:----------|:----------|:----------|:----------|:----------|


##### b. Original Data Analysis
The original data set consists of 39209 different pictures, all of which consists one traffic sign of a certain class.

Since the pictures are of different sizes, the first step is of course to normalize both the picture and object boxes into 32*32*3 matrices. This step is already done by the udacity side. The corresponding data are stored under *./datasets/normalized_data/* directory.
As a result, there are:
1. 34799 training images
2. 4410 validation images
3. 12630 testing images

of size **32\*32\*3** with **43** different label classes, which can be observed directly from the **pyCharm** variable inspector.
Here is an example image which has a bounding box labelled as class **8**:

![][image2]

All 43 classes are listed below:

|       |       |       |       |       |
|:-----:|:-----:|:-----:|:-----:|:-----:|
| ![](./references/classes/c0.jpg) | ![](./references/classes/c1.jpg) | ![](./references/classes/c2.jpg) | ![](./references/classes/c3.jpg) | ![](./references/classes/c4.jpg) |
|0      |1      |2      |3      |4      |
| ![](./references/classes/c5.jpg) | ![](./references/classes/c6.jpg) | ![](./references/classes/c7.jpg) | ![](./references/classes/c8.jpg) | ![](./references/classes/c9.jpg) |
|5      |6      |7      |8      |9      |
| ![](./references/classes/c10.jpg) | ![](./references/classes/c11.jpg) | ![](./references/classes/c12.jpg) | ![](./references/classes/c13.jpg) | ![](./references/classes/c14.jpg) |
|10      |11      |12      |13      |14      |
| ![](./references/classes/c15.jpg) | ![](./references/classes/c16.jpg) | ![](./references/classes/c17.jpg) | ![](./references/classes/c18.jpg) | ![](./references/classes/c19.jpg) |
|15      |16      |17      |18      |19      |
| ![](./references/classes/c20.jpg) | ![](./references/classes/c21.jpg) | ![](./references/classes/c22.jpg) | ![](./references/classes/c23.jpg) | ![](./references/classes/c24.jpg) |
|20      |21      |22      |23      |24      |
| ![](./references/classes/c25.jpg) | ![](./references/classes/c26.jpg) | ![](./references/classes/c27.jpg) | ![](./references/classes/c28.jpg) | ![](./references/classes/c29.jpg) |
|25      |26      |27      |28      |29      |
| ![](./references/classes/c30.jpg) | ![](./references/classes/c31.jpg) | ![](./references/classes/c32.jpg) | ![](./references/classes/c33.jpg) | ![](./references/classes/c34.jpg) |
|30      |31      |32      |33      |34      |
| ![](./references/classes/c35.jpg) | ![](./references/classes/c36.jpg) | ![](./references/classes/c37.jpg) | ![](./references/classes/c38.jpg) | ![](./references/classes/c39.jpg) |
|35      |36      |37      |38      |39      |
| ![](./references/classes/c40.jpg) | ![](./references/classes/c41.jpg) | ![](./references/classes/c42.jpg) |
|40      |41      |42      |      |      |

##### c. Loading the Data

Use the **pickle** library to load the data. The training/validation/testing data are fed to 3 different dictionaries.

##### d. Input Normalization

Since it's the shape, rather than the colour, that actually distinguishes signs from each other, I decide to discard the colour information first to speed up the training.
At the first try, I converted all RGB pictures into grayscale ones
Here is an example:

![][image3]

#### 2) Data Architecture
##### a. **[LeNet](https://en.wikipedia.org/wiki/LeNet)** Structure

![][image_lenet]

##### b. **[Tensorflow](https://en.wikipedia.org/wiki/TensorFlow)** Layered Architecture

My final model consisted of the following layers:

| Layer         		    |     Description	        					| 
|:-------------------------:|:---------------------------------------------:| 
| Input         		    | 32x32x1 greyscale image   					| 
| Layer 1: Convolution 5x5  | 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					    |												|
| Max pooling 2x2      	    | 2x2 stride,  outputs 14x14x6   				|
| Layer 2: Convolution 5x5  | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					    |												|
| Max pooling 2x2      	    | 2x2 stride,  outputs 5x5x16   	    	    |
|-----Flatten Layer-----    | flatten the 5x5x16 input into a 400x1 single layer|
| Layer 3: Fully connected  | outputs 120x1 								|
| Dropout				    |												|
| RELU					    |												|
| Layer 4: Fully connected  | outputs 84x1 	    							|
| Dropout				    |												|
| RELU					    |												|
| Layer 5: Fully connected  | outputs 43x1 logits							|
| Softmax				    |               								|

##### c. The training pipeline
A similar pipeline to that from the course is set to train and valid the model set above.

### 3. Model Training
##### a. First try - rudimentary approach
At the first step, the model was trained using a simple strategy: at a fixed learning rate for certain epochs.
- Learning rate: 0.001
- Batch size: 128
- EPOCHs: 50

As a result, the accuracy is about 88% and still haven't come to convergence.

##### b. Second try - adding some stop criteria
Next, I added the "learning rate decay" technique to my model, with some additional parameters, 
increase the EPOCHs for more training,
and, stop criterias are also added to quit the training at convergence:
- Learning rate initial: 0.005
- Batch size: 128
- EPOCHs: 400
- Learning rate decay steps: 5000
- Learning rate decay rate: 0.9
* Stop when the change in validation accuracy doesn't change more than 0.1% for **5** EPOCHs.

As a result, the accuracy is 91.8%.

I found out that the validation accuracy changes to rapidly during training, so I decided to increase the batch size.

##### c. Third try - larger batch size
The hyperparameters are adjusted as following:
- Learning rate initial: 0.005
- Batch size: 500
- EPOCHs: 400
- Learning rate decay steps: 1000
- Learning rate decay rate: 0.8
* Stop when the change in validation accuracy doesn't change more than 0.1% for **5** EPOCHs.

The accuracy stays roughly the same, except a little lower: 90.6% on the training set

##### d. Fourth try - Add drop out
I tried this time the drop out on the fully connected layer

- Learning rate initial: 0.0025
- Batch size: 256
- EPOCHs: 400
- Learning rate decay steps: 1000
- Learning rate decay rate: 0.8
- Drop out rate: 0.5
* Stop when the change in validation accuracy doesn't chnge more than 0.1% for **10** EPOCHs.

The accuracy improves a little bit: 94.2% on the validation set and 92.2% on the test set.

The training curve is shown below:
![][image_train4]

Cly, there is the problem of **overfitting**, since the accuracy on training data already reaches over 99 percent, whilst the validation accuracy saddles under 95.
Before trying the **data augmentation** technique, I decided to **early stop** the training.

##### e. Fifth try - Early stop

- Learning rate initial: 0.0025
- Batch size: 256
- EPOCHs: 100
- Learning rate decay steps: 1000
- Learning rate decay rate: 0.8
- Drop out rate: 0.5
* Stop when the change in validation accuracy doesn't chnge more than 0.1% for **3** EPOCHs.

With a little luck, I reached the goal this time: 95.9% on the validation set
100% on the training set,
94.7% on the test set.

The training curve is shown below:
![][image_train5]

### 4. Outcomes
My final model results were:
* training set accuracy of 100%
* validation set accuracy of 95.9%
* test set accuracy of 94.7%


#### 1) Evaluation on Test Set

##### a. German traffic signs image

Here are five German traffic signs that I found on the web:

|Nummer| 1     | 2     | 3     | 4     | 5     |
|:---|:-----:|:-----:|:-----:|:-----:|:-----:|
| Picture |![alt text][image_test1] | ![alt text][image_test2] | ![alt text][image_test3] | ![alt text][image_test4] | ![alt text][image_test5]
| Desired Class | 11    | 18    | 25    | 26    | 1     |
| Normalized Picture |![alt text][image_test1_gray] | ![alt text][image_test2_gray] | ![alt text][image_test3_gray] | ![alt text][image_test4_gray] | ![alt text][image_test5_gray]

- Image 1 should be fairly easy to be identified.
- Image 3 has some random background, which might add some difficulties.
- Image 2 is a long graph, which would result in a distorted sign after normalization
- Image 4 also has the similar problem as image 2.
- Image 5 is stretched in another way as image 2 and 4.

##### b. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

##### c. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

##### d. Layer Visualization

#### 2) Potential Improvements