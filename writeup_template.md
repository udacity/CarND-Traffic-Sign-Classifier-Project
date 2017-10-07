#**Traffic Sign Recognition** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

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
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

The project files are included in the zip file as ipynb and as exported html files

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the python and numpy to calculate summary statistics of the traffic
signs data set:




* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is 32,32,3
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? 
I preprocessed the data in 3 ways to avoid a perfect world for the neural network

Real world image normalizations I did
1. Converted to gray scale as color does not "add" additional domain information to what the neural network
needs to perceive, for our domain, this is just noise, this helps redice the input fetures, so that we can
focus on the required output features

2. Normalized the input image pizel values to a zero mean by scaling down range 0 to 255 to -128 to 128
and then further diving by 128 to avoid extremely high values and ranges, now the rage of input data is -1.0 to 1.0
with a mean of 0


3. I did image image augumentation agan as real world images are not perfect. I did 3 augumentations
  a. I did image translation as when the car moves, the image will not be in same camera frame always (used a random scale)
  b. I did image rotation (using a random amount for rotation), same reason, image is not always "perfect picture"
  c. I did image perspective transformation, again for the above sad reasons




 


####2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

After image normalization, I had images of shape 32,32,1
this is the input the processing model/architecture



My final model consisted of the following layers:
The model was pretty much the le net architecture, except I changed the number of featrues to a higher value
because of this paper  http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
Here they are using 100+ features at each stage, so I decided to try this and processed moe features

| Layer         		               |     Description	        					| 
|:-------------------------------:|:---------------------------------------------:| 
| Input         		               | 32x32x1 Grayscale image   							| 
| Convolution 5x5 20 features  	 | 1x1 stride, same padding, outputs 28x28x20 	|
| RELU					                      |												|
| Max pooling	      	            | 2x2 stride,  outputs 14x14x20 				|
| Convolution 5x5 36 features	   | 1 x 1 stride valid padding output 10 x 10 x 36.  
| RELU                           |
| Max Pooling                    | 2x2 stride outputs  5 x 5 x 36
| Flatten                        | outout 900
| Fully Connected                | input 900, output 300
| RELU                           |
| Dropout                        | prob 0.5 during trainging and 1.0 during validation/test
| Fully connected		              | Input 300, output 84
| RELU                           |
| Dropout                        |
| Fully connected                | Input 84, output 43


 


####3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I used a learning rate of 0.000875 so that we do not hit local minima 
I used epoch 13 to 14 as I found out during training , over 14 epochs, learning does not improve, so to prevent 
overfitting < I stopped at 14 epochs. I also used droputs in the model to prevent overfitting
The batch size around 162 has given good results
I stuck with the Adam optimizer as it gave good results

####4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.999
* validation set accuracy of 0.980
* test set accuracy of 0.960 (wow!), ran just once

If an iterative approach was chosen:




* What was the first architecture that was tried and why was it chosen?
I did a hybrid approach, I chose a well known architecture and then fine tuned it
I started with the Le Net architecture in the Lab exercise. I chose this architecture, as it had already proven its
capabilities in winning a championship in recognizing traffic signs.

* What were some problems with the initial architecture?
It did not account for over fitting i.e. did not have drop outs so was training against a perfect world



* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
I then ran the architecture against the raw images provided and got a validation accuracy of 89%
As this was obviously low and did not meet the pass criteria, I started making iterative changes,
the first step was to normaliza the input data i.e. grayscale + mean normalization. This yielded a train accuracy of 
93%. More room for improvement. I then read the link to http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf
which discussed using more features, so I fine tuned the architecure to increase the number of features to 16 from 6 and to 36 from 20. This gave a train accuracy of 95%

Not done yet, so I did more image augumentation i.e. image translation, image rotationa nd image perspective 
changes


* Which parameters were tuned? How were they adjusted and why?
The number of features processed were changed, because the initial architecture had low number of features
compared to whatwas published and tested in the paper http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf

I also changed the epoch to 14 , to avoid over fitting and also lowered the learn rate to 0.000875 from 0.001 
to avoid peaking too soon i.e. local minima


* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Some of the choices were to add drop outs(I added 2 drop ), so to compenate for that I added 1 more fully connected layer
The final result was a stunning 96% on the test set


 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![Exclaim][/Exclaim18.jpeg] ![No Entry][NoEntry.jpeg] ![alt Road Work][RoadWork25.jpeg] 
![Speed Limit][SpeedLimit1.jpeg] ![Stop Sign][StopSign14.jpeg]

The first image might be difficult to classify because ...

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

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

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
####1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


