#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # "Image References"

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./german-traffic-sign-from-web/test1.jpg "Traffic Sign 1"
[image5]: ./german-traffic-sign-from-web/test2_Vorgeschriebene_Fahrtrichtung.jpg "Traffic Sign 2"
[image6]: ./german-traffic-sign-from-web/test3.jpg "Traffic Sign 3"
[image7]: ./german-traffic-sign-from-web/test4.jpg "Traffic Sign 4"
[image8]: ./german-traffic-sign-from-web/test5.jpg "Traffic Sign 5"
[image10]: ./writeup/34798.png "Sample traffic sign"
[image11]: ./writeup/data_distribution.png "Training data distribution"

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I calculated summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 26*25
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is a sample traffic sign displayed.

![Sample traffic sign][image10]

Here is an exploratory visualization of the data set. It is a bar chart showing how the distribution of the training data.

![Data distribution][image11]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to convert the images to greyscale because the traffic signs are designed to be color neutral for recognition.

Here is the code to calculate the greyscale with simple average RGB.

	int(image[i_w][i_h][0]) + int(image[i_w][i_h][1]) + int(image[i_w][i_h][2])/3

I also tried the luminus formula (max(RGB) - min(RGB))/2, but the training result is not that good, so I chose simple average.

As last step, I normalized the image data because it will generate a balanced input and prevent overfitting.

I also generated additional data because the distribution of the training labels is not even. For those occurence < median labels, I rotated the train/validation image with a random angle and append to the train/validation data set, which brought ~6000 new images. This rotation idea is from the original LeNet paper by LeCun. Total ??

I saved the normalized data to binary to save future normalizing time by loading them directly instead of preprocess the original dataset again.

####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:


|      Layer      |               Description                |
| :-------------: | :--------------------------------------: |
|      Input      |         32x32x1 greyscale image          |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 28x28x30 |
|      RELU       |                                          |
|   Max pooling   | 2x2 stride, valid padding, outputs 14x14x30 |
| Convolution 3x3 | 1x1 stride, valid padding, outputs 12X12X60 |
|      RELU       |                                          |
| Convolution 5x5 | 1x1 stride, valid padding, outputs 8x8x120 |
|      RELU       |                                          |
|     Dropout     |                   0.5                    |
|   Max pooling   | 2x2 stride, valid padding, outputs 4X4X120 |
|     Flatten     |               outputs 1920               |
| Fully connected |               outputs 200                |
|      RELU       |                                          |
| Fully connected |                outputs 86                |
|      RELU       |                                          |
| Fully connected |                outputs 43                |
|                 |                                          |
|                 |                                          |



####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an AdamOptimizer for training. With Adam's moving averages of the parameters, it'll take less parameter tuning time.
I tried different batch size from very low (10) to very large (1000), the 1000 is the worst, lower accuracy and slower training time. 10 is medium but is close to 120 actually. Finally I keep 100 as batch size, sth. between 50 - 150 may have similar performance.
For epochs, 40 seems also good, I tried 100 and it performs slightly better, so I use 50 finally. 
For other hyperparameters, I tried sth. a little different, but no big difference, so I keep the original value, mean as 0 and sigma as 0.1.


####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.983 
* test set accuracy of 0.963

I started with classic LeNet, because it's proven to be a good classifier of traffic sign, and there's a good code base built in previous lecture. With initial 2 convolution layers and 3 fully connected layers and unmodified training data set, the test accuracy is ~0.92x. I tried different epoch/batch/hyper parameters, the result kept at similar level. So I decided to try deeper network, one more convolution layer was added, and the depth of each layer was also enlarged, which did improve the test accuracy more. The final change is adding of a dropout layer, which also improved the accuracy. I tried adding it after the 1st layer and 3rd layer, the result is at similar level.
I also modified data set since the distribution of the different signs is not even. Firstly I rotated the images randomly 90 - 270 degree if its label is lower than average. The result is not that good, the test accuracy is higher but not that much. I guess it's because the test data set doesn't inlcude signs rotates that much, which is normal since the real life traffic sign is not going to be rotated 90 degree or so. So I changed the rotation to be less than 15 degree, which did improve the accuracy a lot, validation accuracy to 0.98x and test accuracy to 0.96x.
During the whole process, the validation accuracy is always higher than test accuracy. But when more layers and data are added, the gap between 2 accuracy is less, from initial 0.95x vs. 0.92x, to final 0.98x vs. 0.96x. I think it shows the model is more stable and working well.

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first image might be difficult to classify because there's a number 30 in it. Sometimes it was identified as other number, 80, etc. 
The 2nd one is not a real life sign, it shall be easier for classifier.
The left 3 have more background as noises, which may brings extra difficulties to the model. 

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

|                 Image                 |              Prediction               |
| :-----------------------------------: | :-----------------------------------: |
|         Speed limit (30km/h)          |         Speed limit (30km/h)          |
|           Turn right ahead            |           Turn right ahead            |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
|            General caution            |            General caution            |
|             Priority road             |             Priority road             |


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 96.3%. But during several trial run, the first image with number 30 was identified as other number, 80, etc. And it's also not a complete sign, the lower and right edges are cut off. Adding more perturbed training images shall be helpful.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 13rd cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a 30 km speed limit (probability of 0.99), and the image does contain a 30 km speed limit. The top five soft max probabilities were

|  Probability   |                Prediction                |
| :------------: | :--------------------------------------: |
|      .99       |           Speed limit (30km/h)           |
| 6.90072600e-04 |           Go straight or left            |
| 5.96687132e-05 |           Speed limit (50km/h)           |
| 2.00880404e-05 |             General caution              |
| 1.16388674e-05 | Vehicles over 3.5 metric tons prohibited |


For the second image, the model is relatively sure that this is a Turn right ahead (probability of 0.99). The top five soft max probabilities were

|  Probability   |         Prediction          |
| :------------: | :-------------------------: |
|      .99       |      Turn right ahead       |
| 2.20059675e-07 |    Speed limit (60km/h)     |
| 8.55448121e-08 |    Speed limit (80km/h)     |
| 2.10136464e-09 |    Go straight or right     |
| 1.23743613e-10 | End of speed limit (80km/h) |

For the left 3 images, the top probability is almost 100%, other probabilities are very low, such as 1.27587375e-19. 

However, I did notice in other round of tests, even with same data set and parameters,  the first image was identified as other speed limit, especially the 80km speed limit, it's natural since 3 and 8 looks like very similar. I think adding more relevant training data, or a deeper network/layer depth may increase the accuracy.

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?

Take the con1 as a sample, it's visualized as 30 images. We can find that the CNN identifies the shape of number 30,  and the round circle as features.

It's interesting that one of the image is almost total black, I think it's because the first image's background is almost same color.