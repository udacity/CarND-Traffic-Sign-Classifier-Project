
# Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./sample-training-images.png "Sample training images"
[image2]: ./extra-traffic-images.png "Extra raffic Sign Images"

## Project Home
Here is my [project home at Github](https://github.com/dr-tony-lin/CarND-Traffic-Sign-Classifier-Project.git)

## Data Set Summary & Exploration
The project's data set is summarized as follows:
* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32x32x3
* The number of unique classes/labels in the data set is 42

## Design and Test a Model Architecture

### Preprocessing
The training images are converted into grayscale images first, then the pixels are normalized to have values in the range of [-1, 1].

The following diagram shows 32 images from the training set and their preprocessed counterpart used for training. The images are arranged into 4x8 grids, each cell contains the original image on the top and the preprocessed image at the bottom.

![alt text][image1]

#### Addition Preprocessing Explored 
In addition to the above, I have also experimented the following preprocessing methods, but these did not improve the overall results:

1. Add random noise to the images, and add these images to the original training images that doubled the training set effectively. This is miuch expected as this is one nof the things that convolution achieves.
2. In addition to the original convoulation layer, apply Canny filter and feed the results to the fully connected network layer.

#### Further Work in Preprocessing
The following methods might be explored in the future:
1. Apply random shift, rotation, scale, and other affine transformation to the images
2. Enhance the original images like denoising, color enhancement, contrast ... etc, whether this will be ueful with CNN may be questionable.

### Model Architecture
My final model consisted of 4 convolutional layers, and 3 fully connected network layers. Leaky ReLu is used for activation in the convolutional layers, and tanh is used for activation in the fully connected network layers. This model has been able to achieve test accuracy between 97% to 97.5%.

| Layer         		|     Description	        					        | 
|:---------------------:|:-----------------------------------------------------:| 
| Input         		| 32x32x1 grayscale image   					        | 
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x16 	        |
| Leaky RELU			|												        |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x32 	        |
| Leaky RELU			|												        |
| Convolution 3x3     	| 1x1 stride, same padding, outputs 32x32x64 	        |
| Leaky RELU			|												        |
| Max pooling	      	| 2x2 kernel/stride, valid padding outputs 16x16x64     |
| Convolution 3x3     	| 1x1 stride, valid padding, outputs 14x14x96	        |
| Leaky RELU			|												        |
| Max pooling	      	| 2x2 kernel/stride, same padding outputs 7x7x64        |
| Dropout       		| keep probability: 60%    		                        |
| Fully connected		| 4704x1600, dropout probability: 40%    		        |
| tanh			        |							       				        |
| Dropout       		| keep probability: 60%    		                        |
| Fully connected		| 1600x400,  dropout probability: 40%              		|
| tanh			        |								        		        |
| Dropout       		| keep probability: 60%    		                        |
| Fully connected		| 400x42                    	     			        |
| tanh			        |								        		        |

#### Tensorflow data format
For improving GPU performance, I used *NCHW* format in the model.

#### Initialization of weights
Initialization of weights has been establshed as one key success factor of neural network. I have tried gaussian and xavier initialization, and xavier initialization was able to achieve faster training rate, but after sufficient epochs, there is no observable difference in their training and test accuracy.

#### Activation in Convolution Layers
I have tried ReLu and Leaky ReLu, and Leaky ReLu performs slightly better than ReLu. For this reason, Leaky ReLu is used in my model.

#### Activation in Fully Connected Layer
I have tried ReLu, Leaky ReLu, tanh, and softmax. Tanh performed a lot better than the rest, and is used.

#### Dropout
All the hidden fully connected layers also uses dropout to reduce overfitting. In order to tuen dropout off during perdiction, a placeholder is used. 

#### Alternative Model Experimented
I have tried models with one extra fully connected layer, the repository contains the following models that are among all models that I have tried:

| Folder      	    	|     Description	                   		                                                             | 
|:---------------------:|:------------------------------------------------------------------------------------------------------:|
| CNN-2-4            	| 2 convolution layers, and 4 network layers, gassian weight initialization                              |
| CNN-2-4-local-norm   	| 2 convolution layers with local normalization, and 4 network layers, gassian weight initialization     |
| CNN-3-4           	| 3 convolution layers, and 4 network layers, gassian weight initialization                              |
| CNN-4-3           	| 4 convolution layers, and 3 network layers, gassian weight initialization                              |
| CNN-4-4            	| 4 convolution layers, and 4 network layers, gassian weight initialization                              |
| Gaussian            	| 4 convolution layers, and 3 network layers, gassian weight initialization                              | 
| randomize_image       | 4 convolution layers, and 3 network layers, xavier weight initialization                               | 
| xavier            	| 4 convolution layers, and 3 network layers, xavier weight initialization                               | 

### Training
Training uses Adam Optimizer with a learning rate of 0.0006, a batch size of 256, and 100 epochs for every training. The process tried 10 trainings, and the model that results in the best validation accuracy are stored in checkpoints.

The training was perform on a Winbdows 10 destop with Nvidia GTX 1080. The training time for 1000 epochs was aroung 5600 seconds.
Since Windows 10 does not support GPU pass-through, Docker and Anaconda cannot be used. Fortunately, Python 3.5 worked just fine.

My final model results were:
* Training set accuracy: 99.38%
* Validation set accuracy of 98.5%
* Test set accuracy ranges 97.6%

##### Other optimization experimented
I have also tried Adadelta optimizer, the result was not acceptable. It might be caused by poor tuned learning rate or other hyper parameters. But given the limited time, I could not explore further.

### Test a Model on New Images

Here are 10 German traffic signs that I found on the web, their pre-processed images used for perdiction are also shown:

![alt text][image2]

Here are the results of the prediction, and the probability

| Image			            |     Prediction	        	 | Probability  |
|:-------------------------:|:------------------------------:|:------------:|
| Speed Linit 30km/h        | Speed Linit 30km/h   			 |   15.27%		| 
| Yield     			    | Yield 						 |	 15.27%		|
| Stop					    | Stop							 |	 15.27%		|
| No entry	      		    | No entry  					 |	 15.27%		|
| Dangerous curve to right	| Dangerous curve to right     	 |	 15.26%		|
| Bumpy road            	| Bumpy road                 	 |	 15.27%		|
| Slippery road         	| Slippery road                	 |	 15.23%		|
| Road work             	| Road work                  	 |	 15.24%		|
| Pedestrians             	| Pedestrians                  	 |	 15.26% 	|
| Speed limit 60km/h        | Speed limit 60km/h             | 	 15.24%		|


The model was able to correctly guess 10 of the 10 traffic signs, which gives an accuracy of 100%.

The code for making predictions on my final model is located in the 31th cell of the Ipython notebook, it uses an evaluation model returned from create_softmax_evaluation function defined in cell 6. 

For all of the images, the model is relatively sure (15% vs 2%) when comparing the probability of the predicted label with the rest of the labels.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
TODO later as I spent most my available time on the models and tests
