# **Traffic Sign Recognition** 

## Writeup

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

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/khaemn/CarND-Traffic-Sign-Classifier-Project/blob/develop/Traffic_Sign_Classifier-TF2-GPU.ipynb)

The HTML saved from the full run can be found [here](https://github.com/khaemn/CarND-Traffic-Sign-Classifier-Project/blob/develop/saved_html/Traffic_Sign_Classifier-TF2-GPU.html)


### Data Set Summary & Exploration

#### 1. Basic summary of the data set. In the [code](./Traffic_Sign_Classifier-TF2-GPU.ipynb), the analysis is done using python, numpy and/or pandas methods.

[barplot_training]:   ./images/barplot_training.png "Barplot distribution"
[barplot_validation]: ./images/barplot_validation.png "Barplot distribution"
[barplot_testing]:    ./images/barplot_testing.png "Barplot distribution"


I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799 examples
* The size of the validation set is 12630 examples (about 36% of training)
* The size of test set is 4410 examples (about 13% of training)
* The shape of a traffic sign image is 32x32 pixels, 3 color channels
* The number of unique classes/labels in the data set is 43

#### 2. Exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. 

|![][barplot_training]|![][barplot_validation]|![][barplot_testing]|
|:-------------------:|:---------------------:|:------------------:| 

As it is shown above, all the train, validation and test splits in GTSRB dataset have similar distributions. The most common classes are for signs '50_speed', '30_speed', 'give_way', and the rarest is 'attention_left_turn'. I suppose, that the model accuracy would be the best for most common signs and the poorest for the rarest ones.

Also I have noticed there is no 'no sign' or 'empty' class, so I suppose any model, trained on this dataset, would treat *any* image as a road sign and providing some non-zero probability for it. This should be taken into account when deploying the model.

Probably, a good investment would be to add some random images and/or noise to a special "empty" class, having the size of this class about 1/43 (TBD) of the total examples in the dataset - in this case the model might also detect that there is NO road sign in a given image.


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. Describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.

[normalization]:           ./images/normalization.png "Normalization"
[augmentation_techniques]: ./images/augmentation_techniques.png "Augmentation options"
[augmentation]:            ./images/augmentation.png "Augmented training imgs"

As a first step, I decided to convert the images to grayscale, because it might save some model weights three times, as working with only 1 channel requires less resources. However, a couple of quick runs on a random 10-15% of the training dataset showed that the colored (3-channel) images works better in terms of the validation accuracy. I suppose, the color information (red circles, blue circles, ... ) is really important for recognizing signs, which are poorly visible. In fact, any (human) driver is tested for a clear color vision in particular; so I believe the grayscaling is not an option here.

I normalized the image data using subtraction and division to make any image fit into [-0.5..0.5] range. This is required for further processing with a neural network, as the network works poorly with values greater then [-1..1]. This is done with `normalize_pickle()` function in the notebook.

To plot images, I use a helper script `plot_helper.py` (left from the previous lane detection project). For example, the function `to_image()` can convert a float or integer 2-D array to an OpenCV-compatible numpy array of uint8 in range [0..255]. It is very convenient for plotting normalized images or other arrays from various processing steps.

Below is an example of a validation image #1010, directly plotted from the training data (left), and after normalization and scaling back to range [0..255] via `to_image()` function (right):

![][normalization]

I decided to generate additional data because I have nod found much images with a visual noise in the training dataset, and I suppose that the real pictures could be worse. So I have implemented several functions for augmentation: salt&pepper noise, bleach, zoom in and grayscaling. Below is shown the original validation image #1010 (left), a 3 types of color bleach in the top row, and other types of augmentation in the bottom row:

![][augmentation_techniques]

To generate the augmented dataset, I used combinations of these techniques. The original image (top left) and examples of 7 augmentations, used for the dataset, are shown below:

![][augmentation]

After augmentation, each of the images were normalized. Normalization only makes sense _after_ augmentation, because all my augmenting functions works with a regular [0..255] uint8 image.

My augmented training dataset contains 278392 images (about 7 times more then the original one).


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

|       Layer type    | Input shape |     Description	               | Output shape |
|:---------------------:|:--------:|:-------------------------------------:|:--------:| 
| Input         		| 32x32x3  | RGB image   		                   | 32x32x3  |
| Convolution 1 5x5     | 32x32x3  | 1x1 stride, valid padding, 16 filters | 28x28x16 |
| Max pooling 1 3x3     | 28x28x16 | 2x2 stride, valid padding             | 26x26x16 |
| Convolution 2 5x5     | 26x26x16 | 1x1 stride, valid padding, 16 filters | 22x22x16 |
| Max pooling 2 3x3     | 22x22x16 | 2x2 stride, valid padding             | 10x10x16 |
| Convolution 3 5x5     | 10x10x16 | 1x1 stride, valid padding, 16 filters | 6x6x64   |
| Max pooling 3 3x3     | 6x6x64   | 2x2 stride, valid padding             | 2x2x64   |
| Flatten               | 6x6x64   |                                       |   256    |
| Fully connected 1		| 256      | Classes * 2 == 86 neurons             |   86     |
| Fully connected 2		| 86       | Classes * 3 == 126 neurons            |   126    |
| Fully connected 3		| 256      | Classes == 43 neurons                 |   43     |
| Softmax				| 43       |     								   |   43     |

 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the LeNet network as a point to start. The first network I trained was the classical LeNet with 2 Convolution and 2 Dense layers (the last one had 43 outputs instead of 10, of course). 
However, the validation accuracy has never reached above 0.06. So I have run over the lectures again, and realized that the weight/bias initialization was wrong - just a random floating point numbers, but not the `truncated_normal()`. Making the proper intialization has immediately improved the accuracy. After some experiments, I managed to reach the validation accuracy about 0.94. 

Then I considered the LeNet was developed to work with a grayscale images, and I am using 3-channel, which obviously have more information embedded; and also the number of classes is 4 times more. So, I have added some filters to the first Conv layer (6 --> 8) and also added one more Dense layer. To prevent overfitting, I have used Dropout after each MaxPool layer (as I remember, I saw this approach the first time in some Keras manuals). After a couple experiments with learning rate, batch size and dropout rate, I understood that running the training on CPU does not work good for me, as it is quite slow.

I have spent a lot of effort trying to set up GPU runtime using Tensorflow 1.13, 1.12 and 1.14, and various versions of CUDA and CuDNN libraries, however, without a success. Neither I wanted to run the experiments in the project environment, as it is not quite convenient for me.

So I have swithced to using Tensorflow 2.3.1, and that required some changes to the code, however I tried to keep it as close to the original lectures as possible. Hope that wouldn't be a problem.

While training on GPU (I have NVidia MX250 with 384 CUDA cores laptop version), I had about 10 times performance boost. Still, most the experiments with the learning rate, batch size, dropout rate and additional layers I performed on a fraction of the augmented training dataset. It turned out I can use 1/20..1/8 of the whole augmented dataset to quickly check if any change to a hyperparameter improves the validation accuracy or not.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


Totally I made about 50 experiments with various hyperparameters, some variations of the augmentation pipeline, and some architecture changes.


My final model results were:
* training set accuracy of 0.985
* validation set accuracy of 0.977 
* test set accuracy of 0.955

I have chosen an iterative approach, because I wanted to achieve the best possible result.

###### What was the first architecture that was tried and why was it chosen?
Initially, it was a regular LeNet architecture

###### What were some problems with the initial architecture?
As the LeNet is intended to be used with grayscale image, and I understood that the color information for the road signs is valuable and should not be discarded, the issue was: how to make LeNet work with colored images. At first, I just added more channels to the input convolution layer.

###### How was the architecture adjusted and why was it adjusted? 
The resulting validation accuracy of the very first architecture was relatively low (<0.75), so I was adding Conv2D filters (e.g. increasing the output depth of a Conv layer), then adding more Conv2D-Maxpool chains. As I have bumped into overfitting (detected by having the training accuracy >0.9 and validation accuracy <0.7), I have added a Dropout layers, right after Pooling.
I have also tried to vary Conv layers filter size (3x3, 7x7 and their combinations, e.g. different filter sizes in different layers), but the validation accuracy increase was either not reached, or just was not worth the performance drop.

###### Which parameters were tuned? How were they adjusted and why?
I have adjusted dropout layers quantity (1 .. 3) and the dropout rate (0,1 .. 0,5), convolution filter sizes (3x3, 5x5, 7x7) and convolution layers quantity (2 .. 3), maxpooling window size (2x2, 3x3).
Also I have experimented with various augmentation techniques.
To prevent overfitting and to automatize the training process, I have implemented an early stop algorithm, which took the best (maximum) validation accuracy, and if in next 5 epochs a better result was not reached, stopped the training. Using this approach, I had just set epoch counter to some large number, and let the TensorFlow to do the rest.

###### What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
3 dropout layers with rate about 0.2 have helped a lot against overfitting (e.g. having a good accuracy on the training dataset but poor accuracy on the validation one).


### Test a Model on New Images

[sign_01]: ./signs_in_a_wild/crop_resize/001-limit_50-resized.jpg "Limit 50"
[sign_02]: ./signs_in_a_wild/crop_resize/002-priority-resized.jpg "Priority"
[sign_03]: ./signs_in_a_wild/crop_resize/003-both_ways-resized.jpg "Both ways"
[sign_04]: ./signs_in_a_wild/crop_resize/004-construction-resized.jpg "Construction"
[sign_05]: ./signs_in_a_wild/crop_resize/005-limit-70-resized.jpg "Limit 70"
[sign_06]: ./signs_in_a_wild/crop_resize/006-give-way-resized.jpg "Give way"
[sign_07]: ./signs_in_a_wild/crop_resize/007-pedestrian-resized.jpeg "Pedestrian"
[sign_08]: ./signs_in_a_wild/crop_resize/008-no-left-turn-sign-resized.jpg "No left turn"
[sign_09]: ./signs_in_a_wild/crop_resize/009-straight_ahead-resized.jpeg "Straight ahead"
[sign_10]: ./signs_in_a_wild/crop_resize/010-stop-resized.jpg "STOP"


#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

|                    |                    |                    |
|:------------------:|:------------------:|:------------------:|
|![alt text][sign_01]|![alt text][sign_02]|![alt text][sign_03]|
|![alt text][sign_04]|![alt text][sign_05]|![alt text][sign_06]|
|![alt text][sign_07]|![alt text][sign_08]|![alt text][sign_09]| 
|                    |![alt text][sign_10]|                    |

In the training dataset, there were no sign "both ways" and "no left turn", so I has expected the model to fail on prediction of these signs. Other ones were present in the dataset while training, so I anticipated them to be recognized correctly.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Limit 50 km/h    		| 50_speed   									| 
| Priority     			| right_of_way_general  						|
| Both ways				| attention_children   **INCORRECT**			|
| Construction    		| construction_ahead			 				|
| Limit 70 km/h			| 70_speed         	   **INCORRECT**			|
| Give way      		| give_way   									| 
| Pedestrian    		| turn_straight_left   **INCORRECT**   			| 
| No left turn    		| 70_speed  									| 
| Straight ahead   		| turn_straight									| 
| STOP          		| Stop sign   									| 


The model was able to correctly guess 7 of the 10 traffic signs, which gives an accuracy of 70%. This compares fairly to the accuracy on the test set of 0.95 (95%), especially taking into account that some signs are expected to be incorrectly recognized. If compute the accuracy only on 8 signs that are present in training dataset, the model shows 7/8 (0.875) accuracy in this test.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability.

The code for making predictions on my final model is located in the **"Output Top 5 Softmax Probabilities For Each Image Found on the Web"** section of the Ipython notebook.

Actually, only one example ("Pedestrian") has the expected probability distribution:

```
007-pedestrian-resized.jpeg :
   0.923 : turn_straight_left
   0.025 : turn_circle
   0.020 : turn_left_down
   0.017 : turn_right
   0.013 : attention_bumpers
```

while the others just have close to 1.00 probability for the top-probably sign, and almost 0.0 for others. I guess it might be related to that I haven't used a softmax at the Model's end while training. Actually, such a high certainty (especially for the signs that were not present in the training dataset) is not a good result, because it could be caused by a high overfitting.

As the model still shows acceptable accuracy on the test dataset, I believe it should not be a problem. However, for a production-grade model, an additional investigation is necessary, and, probably, retraining of a smaller model.



### Visualizing the Neural Network

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

[features_01]: images/visualized_features.png "Visualized features for STOP sign"
[sign_10]: ./signs_in_a_wild/crop_resize/010-stop-resized.jpg "STOP"

Below, a visualized features for a "STOP" sign image are presented. Clearly, the convolutional filters highlight some low-level features, as the horizontal and vertical straight lines, lines with angle about 45 degrees, or a solid color regions. I presume, the next convolution filters extract more complex features, such as shapes.

For this stop sign there's obviously a vertical and horizontal short lines and also a short lines with angle of 45 degrees found on the image (feature maps 1, 4, 6, 9, 12, 15), which together can be classified as an octagonal shape - and as we know, an octagonal shape is a noticeable feature of the "STOP" sign (and only of this sign). Also, there is a solid color region in the centre of the sign found and highlighted (see feature maps 10 and 11), I presume it might be a red color. There is no feature map which would highlight only the text in the center, but probably maps 9 and 7 together might highlight the text itself. So, logically, an octagonal sign, that is filled with red color and has a text inside, is probably a stop sign.

![alt text][sign_10]

![alt text][features_01]


