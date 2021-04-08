# **Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./preproc.png
[image4]: ./test_data/14_stop.jpg "Stop Sign"
[image5]: ./test_data/17_no_entry.jpg "No Entry"
[image6]: ./test_data/23_Slippery_road.jpg "Slippery Road"
[image7]: ./test_data/35_Ahead_only.jpg "Ahead Only"
[image8]: ./test_data/40_Roundabout.jpg "Roundabout"
[screenshot1]: ./Train_Dataset_counts.png
[screenshot2]: ./Valid_Dataset_counts.png
[screenshot3]: ./Test_Dataset_counts.png
[valid_acc]: ./validation_acc.png
[1st_reslt_acc]: ./result_1st_architecture.png
[2nd_reslt_acc]: ./result_2nd.png
[3rd_reslt_acc]: ./result_3rd.png
[4th_reslt_acc]: ./result_4th.png
[new_img_reslt]: ./result_new_image_with_prob.png
[new_img_bar]: ./result_new_image_with_barChart.png

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the "numpy" library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how many samples are included in the data set for train, valid and test as well.

![alt text][screenshot1]
![alt text][screenshot2]
![alt text][screenshot3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because several images in the training were pretty dark and contained only little color.  
I judged that the grayscaling could reduces the amount of features and thus reduce execution time. Additionally, several research papers have shown good results with grayscaling of the images.

Then, I normalized the grayscaled data using formular (pixel - 128)/128 which converts the int values of each pixel [0, 255] to float values with range [-1, 1]. I've read it helps in speed of training and performance because of things like resources.

Here is an example of an original image and an augmented image:

![alt text][image2]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The model architecture is based on the LeNet model. I added dropout layers before each fully connected layer in order to prevent overfitting. 
My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 26x26x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 				    |
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16   |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 				    |
| Flatten				| outputs 400									|
| Dropout				|           									|
| Fully connected		| outputs 120  									|
| RELU					|												|
| Dropout				|           									|
| Fully connected		| outputs 84 									|
| RELU					|												|
| Dropout				|           									|
| Fully connected		| outputs 43 									|
| Softmax				| etc.        									|
|						|												|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.  

To train my model, I used an 'Adam optimizer' and the following hyperparameters as below.

batch size: 128  
number of epochs: 100  
learning rate: 0.0005  
Variables were initialized using the truncated normal distribution with mu = 0.0 and sigma = 0.1  
keep probalbility of the dropout layer: 0.5  

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.
  
#### < Results >
My final model results were:  
* training set accuracy of 0.991
* validation set accuracy of 0.961  
![alt text][4th_reslt_acc]
* test set accuracy of 0.941

#### < Solution (approach) >
* I used an iterative approach for the optimization of validation accuracy based on Lenet model Architecture.  
The first approach I chosen is original Lenet model Architecture from the Udacity course.  
(used hyper parameters: EPOCHS=10, BATCH_SIZE=128, learning_rate = 0,001, mu = 0, sigma = 0.1)  
below result is the validation accuracy.  
![alt text][1st_reslt_acc]  
Although validation accuracy tends to improve, it's not complete to satisfy classifying traffic signs in the test. I thought it is caused from too small the epochs.  
The second approach, I increased the epochs upto 100.  
![alt text][2nd_reslt_acc]  
At this time validation accuracy seems to be saturation, but there was an overfitting in test accuracy since 25 epochs. (validation accuracy = 0.933, test accuracy = 0.915)  
The third approach, I added dropout layer for reducing overfitting.  
![alt text][3rd_reslt_acc]  
It looks good.  (validation accuracy = 0.978, test accuracy = 0.954)  
The last approach, I tuned learning rate to 0.0005.  
![alt text][4th_reslt_acc]  
(validation accuracy = 0.961, test accuracy = 0.941) 

---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] : 14, Stop  
![alt text][image5] : 17, No entry  
![alt text][image6] : 30, Beware of ice/snow  
![alt text][image7] : 35, Ahead Only  
![alt text][image8] : 40, Roundabout Mandatory  

I thought all images would be classified without a big issues because the shape of those traffic signs and the symbols are relatively characteristic.  

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

#### < Results >  
Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| No-entry    			| No-entry      								|
| Beware of ice/snow	| Beware of ice/snow							|
| Ahead Only      		| Ahead Only    				 				|
| Roundabout Mandatory	| Roundabout Mandatory							|
|                       |                                               |


#### < Analysis >    
The model was able to correctly guess all(5/5) traffic signs, which gives an accuracy of 100%. 
This compares favorably to the accuracy on the test set of about 94%.
If I were to choose one image that was hard to classify, the "Beware of ice/snow" sign. The reason why is that the triangular shape is similar to several other signs in the data set. (Children crossing or Bicycles crossing)

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

#### < Results with probability >
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 100%         			| Stop sign   									| 
| 100%     				| No-entry 										|
| 98.74%				| Beware of ice/snow							|
| 91.07%      			| Ahead Only   				     				|
| 79.51%			    | Roundabout Mandatory	   						|

![alt text][new_img_reslt]
![alt text][new_img_bar]

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


