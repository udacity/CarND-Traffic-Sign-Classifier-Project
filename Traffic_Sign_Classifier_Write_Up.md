#**Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/traffic_sign_original.png "Original Image"
[image2]: ./examples/traffic_sign_pp.png "After Preprocessing"
[image3]: ./examples/traffic_sign_aug.png "After Random Affine Transformation"
[image4]: ./examples/onlineimage_0.png "Traffic Sign 1"
[image5]: ./examples/onlineimage_1.png "Traffic Sign 2"
[image6]: ./examples/onlineimage_2.png "Traffic Sign 3"
[image7]: ./examples/onlineimage_3.png "Traffic Sign 4"
[image8]: ./examples/onlineimage_4.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  For the "stand out" suggestions in the rubic, I include "Augment the training data", "create visualizations of the softmax probabilities", "visualize layers of the neural network". The soft max probabilities I got are almost one for one class and zero for the rest and I did not consider plotting them. 

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

My final version of the code for submission is the [notebook](https://raw.githubusercontent.com/dhuangdeveloper/CarND-Traffic-Sign-Classifier-Project/master/Traffic_Sign_Classifier_Submission.ipynb)
The write-up is the [Markdown document](https://github.com/dhuangdeveloper/CarND-Traffic-Sign-Classifier-Project/blob/3_18_2017/Traffic_Sign_Classifier_Write_Up.md)

###Data Set Summary & Exploration(https://github.com/dhuangdeveloper/CarND-Traffic-Sign-Classifier-Project)

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in the 3th code cell of the IPython notebook.  

I used the numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is 32 x 32 x 3 (RGB)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

Here is an exploratory visualization of the data set. It is a code cell 6 of the notebook. From each class, one random selected image is displayed together with title. 

We observe that 

- The brightness varies significantly among the images, and some of the images are very dark such as image (2,2), image (3,4). (i,j) is the subplot grid.
- Some traffic signs are slightly rotated, e.g., image (3,2), image (6,3).
- The traffic signs are not necessarily centered.
- The traffic signs that should be of the same physical size are of different size in the image. For example, it is reasoanble to expect that the traffic signs in image (1,1) and image (1,2) are of the same physical size. 

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The defintion of the processing function for this step is contained in the 9th code cell of the IPython notebook. The preprocessed images are visualized in code cell 11.

The preprocessing did was to normalize / rescale the V channel of the HSV representation, and is consisted of 3 steps:

- Step1: Convert RGB to HSV representation
- Step2: Rescale the intensiy of the V channel.
- Step3: Convert HSV representation back to RGB.

The output of the preprocessing step is still RGB. The reason behind this is to compensate for the different brightness of the image. Some of the traffice signs (such as image (3,4) in code cell 11) looks better than the same image before preprocessing (image (3,4) in code cell 6)


Here is are 43 images (one from each class) before and after preprocessing.

![alt text][image1]
![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

I increase the training data set by random affine transform, defined in code cell 12. Each image is transformed by the combination of the following transformations:

- Rotate by a random degree between [-8, 8]
- Scale by a random factor between [0.9, 1.1]
- Translate in x and y direction by a random pair in [-2,2] x [-2,2]

For each original image in training, 10 randomly transformed images are generated. (See code cell 14). Note that the original image is NOT included.

My final training data set is of size 347990.

The first set of images after the random affine transformation is visualized in code cell 14.
Here is are 43 images after transformation.
![alt text][image3]

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in the 18th cell of the ipython notebook. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 1x1x3    	| linear combination across color channels		|
| Convolution 3x3    	| 1x1x1 stride, valid padding, outputs 30x30x4 	|
| RELU					|												|
| Convolution 3x3    	| 1x1x1 stride, valid padding, outputs 28x28x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x16 				|
| Convolution 3x3    	| 1x1x1 stride, valid padding, outputs 12x12x32	|
| RELU					|												|
| Max pooling	      	| 3x3 stride,  outputs 4x4x32     				|
| Convolution 3x3    	| 1x1x1 stride, valid padding, outputs 2x2x64	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 1x1x64     				|
| Fully connected		| input 64, output 100        					|
| RELU					|												|
| Fully connected		| input 100, output 43        					|
| Softmax				|           									|



####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in the 23th cell of the ipython notebook. 

To train the model, I used 200 EPOCHS, and BATCH SIZE of 64, and AdamOptimizer with learning rate 0.001. It turns out the validation error stops decreasing after around 100 EPOCHS.

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in the 26th cell of the Ipython notebook.

My final model results were:
* training set accuracy of 98.8%
* validation set accuracy of 97.3%
* test set accuracy of 94.7%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
The first architecture chosen was Lenet with grayscale image because it was proven architecture for similar problems.
I then experimented with different way of working with color channels and different number of covolutional layers, guided by the two principles: Try to let the network to decide all the coefficients, and reduce the number of parameters. 
I went through a trial and error process where I modified lenet or later architecture by a change suggested by the above principles. 

A few different schemes I work through include:
1. Combining colors: a few options I experimented with include:
   1.a: known RGB to grayscale weight
   1.b: convert to grayscale with coefficieint trained as part of the network
   1.c: Using rgb channel
   1.d: Combing rgb into 2 channels with coefficient trained as part of the network
2. Number of convolutional layers and NN layers:
   To reduce the number of parameters, I want to reduce the number of final outoput channels from final CNN layer so that my NN layers remain simple, and I also want to increase the number of CNN layers so that I explore more combinations of filters while keeping the number of coefficients in the CNN layers small. My final CNN is 5 layers (the first layer is not a really a filter on 2D image but rather a combination of color channels). 

Note: I have not experimented with dropout. I meet a pratical constraint that I have a limited amount of budget to spent on AWS GPUs.
 
###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The 4th image might be difficult to classify: The 4th image is a double curve but the example of the double curve might be of two different directions  (can be drown as |\| or |/|).

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in the tenth cell of the Ipython notebook.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Children crossing		| Children crossing								| 
| Pedestrians  			| Detection Road narrows on the right			|
| Speed limit (20km/h)	| Speed limit (20km/h)							|
| Double curve     		| Road work 					 				|
| Keep right			| Keep right        							|


The model was able to correctly guess 3 of the 5 traffic signs, which gives an accuracy of 60%. 

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 34 cell of the Ipython notebook.

For the first image, the model is almost sure that this is a Children crossing (probability of 1.00), and the image is a Children crossing. 

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.00         			| Children crossing								| 
| 0.00     				| Road narrows on the right						|
| 0.00					| General caution								|
| 0.00	      			| Road work 					 				|
| 0.00				    | Bicycles crossing    							|

For the second image, the model is almost sure (wrongly) that this is a road narrows on the right. The second highest probaiblty (but close to 0) is Pedestrian, which is the correct answer. All top 5 categories are triangle shaped signs. 

For the third image, the model is highly confident (0.87) that this is a speed limit (200km/h) which is the correct answer. Interestingly, all the top 5 are speek limit signs with different speed limit.

For the fourth image, the model thinks this is most likely (0.66)  road work sign while the actual answer is Double Curve. 

For the 5th image, the model thinks this is almost sure that this is a keep right sign which is correct.

