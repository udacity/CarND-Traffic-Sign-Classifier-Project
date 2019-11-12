# **Traffic Sign Recognition** 

## Writeup


The goals / steps of this project are the following:
* Load the data set (provided)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output/sample_class_files.png "Visualization"
[image2]: ./output/original_gray.png "Grayscaling"
[image3]: ./output/train_class_histogram.png "Histogram"
[image4]: ./output/new_images.png "New Images"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The project has been done in the provided workspace.
The workspace shall be saved and provided as HTML file as specified as the project requirement.

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The training, validation and testing files are loaded using pickle. 

* The size of training set is **34799**
* The size of the validation set is **4410**
* The size of test set is ? **12630**
* The shape of a traffic sign image is ? **(32,32,3)**
* The number of unique classes/labels in the data set is ? **43**

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It sampled one of the image files per each class to see how each class file looks. 
![alt text][image1]

The bar chart is showing how many training image files are available for each class/label for training data.
![alt text][image3]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to grayscale because it is easier to convert for the normalization and the model being used expects one color channel.

Here is an example of a traffic sign image before and after grayscaling, for example.

![alt text][image2]

As a last step, I normalized the image data, so that the data has mean zero and equal variance. The method is (pixel - 128)/ 128 as suggested, which is a quick way to approximately normalize the data.


// optional. The difference between the original data set and the augmented data set is the following ... 


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale/normalized image   			| 
| Convolution 5x5     	| 5x5 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			    	|
| Convolution 5x5	    | 5x5 stride,  outputs 10x10x16      			|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			    	|
| Flatten       		| Input 5x5x16, outputs 400						|
| Fully connected		| Input 400, outputs 120     					|
| RELU					|												|
| Fully connected		| Input 120, outputs 84     					|
| RELU					|												|
| Fully connected		| Input 84, outputs 43     			    		|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I tried several different hyperparameters (learning rate, epoch, batch size) to see which combination provides the best accuracy.
The training/validation test output with these different hyperparameters settings are as follows. 


**Learning rate based test**

rate = 0.001
EPOCH 250 ...
Validation Accuracy = 0.949
Test Accuracy = 0.933

rate = 0.005
EPOCH 250 ...
Validation Accuracy = 0.938
Test Accuracy = 0.919

rate = 0.01
EPOCH 250 ...
Validation Accuracy = 0.083
Test Accuracy = 0.078

----------------------
**BATCH size based test**

BATCH_SIZE = 60 

EPOCH 250 ...
Validation Accuracy = 0.937
Test Accuracy = 0.930

BATCH_SIZE = 200 

EPOCH 250 ...
Validation Accuracy = 0.938
Test Accuracy = 0.927

----------------------

**EPOCH based test**

EPOCH 5 ...
Validation Accuracy = 0.894
Test Accuracy = 0.882

EPOCH 20 ...
Validation Accuracy = 0.926
Test Accuracy = 0.908

EPOCH 50 ...
Validation Accuracy = 0.942
Test Accuracy = 0.922

EPOCH 70 ...
Validation Accuracy = 0.929
Test Accuracy = 0.909

EPOCH 100 ...
Validation Accuracy = 0.947
Test Accuracy = 0.925

EPOCH 150 ...
Validation Accuracy = 0.938
Test Accuracy = 0.930

EPOCH 200 ...
Validation Accuracy = 0.946
Test Accuracy = 0.930

EPOCH 250 ...
Validation Accuracy = 0.949
Test Accuracy = 0.933

EPOCH 300 ...
Validation Accuracy = 0.950
Test Accuracy = 0.929


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of ? **0.937**
* test set accuracy of ? **0.930**

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] 

The first image might be difficult to classify because ...

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed limit (20km/h) 	| Roundabout mandatory							| 
| Right-of-way at the next intersection	| Right-of-way at the next intersection			|
| Yield 				| Yield											|
| No entry				| No entry  									|
| Road narrows on the right   | Road narrows on the right 				|
| End of all speed and passing limits   | Speed limit (30km/h)          |
| Keep left     		| Slippery Road      							|
| Roundabout mandatory  | Roundabout mandatory                          |

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. The accuracy on these new images are lower than the accuracy achieved on the test set.
It could be that the class 0 and class 32 images are under fitted to improve the accuracy on the test images of these classes.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 8th cell of the Ipython notebook.

For the first image(**Speed limit (20km/h)**), the model predicted <span style="color:red">wrongly</span>. The top five soft max probabilities were as follows.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .51         			| Speed limit (70km/h)   						| 
| .49     				| Turn right ahead	                    		|
| .0007					| Speed limit (30km/h)							|
| .00005	   			| General caution					 			|
| .00000005			    | Speed limit (100km/h)          				|


For the second image(**Right-of-way at the next intersection**), the model predicted correctly. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .99         			| Right-of-way at the next intersection     	| 
| .00001   				| Beware of ice/snow	                  		|
| .00000000001			| Pedestrians							        |
| .0000000000005		| Children crossing					 			|
| .0000000000000008     | Roundabout mandatory          				|

For the third image(**Yield**), the model predicted correctly. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Yield     	                                | 
| .0       				| Speed limit (20km/h)	                  		|
| .0	        		| Speed limit (30km/h)					        |
| .0		            | Speed limit (50km/h)					 		|
| .0                    | Speed limit (60km/h)          				|

For the fourth image(**No entry**), the model predicted correctly. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| No entry     	                                | 
| .0       				| Turn right ahead	                       		|
| .0	        		| Turn left ahead				                |
| .0		            | Roundabout mandatory					 		|
| .0                    | Slippery road          				        |

For the fifth image(**Road narrows on the right**), the model predicted correctly. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Road narrows on the right                     | 
| .0       				| Traffic signals	                       		|
| .0	        		| Pedestrians				                    |
| .0		            | Speed limit (70km/h)					 		|
| .0                    | General caution        				        |

For the sixth image(**End of all speed and passing limits**), the model predicted <span style="color:red">wrongly</span>. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed limit (30km/h)                          | 
| .0       				| Roundabout mandatory	              		    |
| .0	        		| Speed limit (50km/h)		                    |
| .0		            | Speed limit (60km/h)					 		|
| .0                    | Stop      				                    |

For the seventh image(**Keep left**), the model predicted wrongly. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep left                                     | 
| .0       				| Turn right ahead	                  		    |
| .0	        		| Speed limit (50km/h)		                    |
| .0		            | Speed limit (30km/h)					 		|
| .0                    | Stop      				                    |

For the eigth image(**Roundabout mandatory**), the model predicted wrongly. The tope five soft max probabilities were as follows.
| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Keep left                                     | 
| .0       				| Speed limit (30km/h)	               		    |
| .0	        		| Priority road		                            |
| .0		            | Speed limit (20km/h)					 		|
| .0                    | Speed limit (50km/h)    




