

# Traffic signs classification with a convolutional network

This is my attempt to tackle traffic signs classification problem with a convolutional neural network implemented in TensorFlow (reaching **99.33% accuracy**). The highlights of this solution would be data preprocessing, data augmentation, pre-training and skipping connections in the network. You can read [the full post here](http://navoshta.com/traffic-signs-classification/).

Classification of German traffic signs is one of the assignments in Udacity Self-Driving Car Nanodegree program, however the dataset is publicly [available here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset).

## Dataset

The [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset) consists of **39,209 32×32 px color images** that we are supposed to use for training, and **12,630 images** that we will use for testing. Each image is a photo of a traffic sign belonging to one of 43 classes, e.g. traffic sign types.

Each image is a 32×32×3 array of pixel intensities, represented as `[0, 255]` integer values in RGB color space. Class of each image is encoded as an integer in a 0 to 42 range. Dataset is very unbalanced, and some classes are represented way better than the others. The images also differ significantly in terms of contrast and brightness, so we will need to apply some kind of histogram equalization, this should noticeably improve feature extraction.

## Preprocessing

The usual preprocessing in this case would include scaling of pixel values to `[0, 1]` (as currently they are in `[0, 255]` range), representing labels in a one-hot encoding and shuffling. Looking at the images, histogram equalization may be helpful as well. We will apply _localized_ histogram equalization, as it seems to improve feature extraction even further in our case. 

I will only use a single channel in my model, e.g. grayscale images instead of color ones. As Pierre Sermanet and Yann LeCun mentioned in [their paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf), using color channels didn't seem to improve things a lot, so I will only take `Y` channel of the `YCbCr` representation of an image.

## Augmentation

The amount of data we have is not sufficient for a model to generalise well. It is also fairly unbalanced, and some classes are represented to significantly lower extent than the others. But we will fix this with data augmentation!

### Flipping

First, we are going to apply a couple of tricks to extend our data by _flipping_. You might have noticed that some traffic signs are invariant to horizontal and/or vertical flipping, which basically means that we can flip an image and it should still be classified as belonging to the same class. Some signs can be flipped either way — like **Priority Road** or **No Entry** signs, other signs are *180° rotation invariant*, and to rotate them 180° we will simply first flip them horizontally, and then vertically. Finally there are signs that can be flipped, and should then be classified as a sign of some other class. This is still useful, as we can use data of these classes to extend their counterparts. We are going to use this during augmentation, and this simple trick lets us extend original **39,209** training examples to **63,538**, nice! And it cost us nothing in terms of data collection or computational resources. 

### Rotation and projection

However, it is still not enough, and we need to augment even further. After experimenting with adding random *rotation*, *projection*, *blur*, *noize* and *gamma adjusting*, I have used *rotation* and *projection* transformations in the pipeline. Projection transform seems to also take care of random shearing and scaling as we randomly position image corners in a `[±delta, ±delta]` range.

## Model 

### Architecture

I decided to use a deep neural network classifier as a model, which was inspired by [Daniel Nouri's tutorial](http://navoshta.com/facial-with-tensorflow/) and aforementioned [Pierre Sermanet / Yann LeCun paper](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). It is fairly simple and has 4 layers: **3 convolutional layers** for feature extraction and **one fully connected layer** as a classifier.

As opposed to usual strict feed-forward CNNs I use **multi-scale features**, which means that convolutional layers' output is not only forwarded into subsequent layer, but is also branched off and fed into classifier (e.g. fully connected layer). Please mind that these branched off layers undergo additional max-pooling, so that all convolutions are proportionally subsampled before going into classifier.

### Regularization

I use the following regularization techniques to minimize overfitting to training data:

* **Dropout**. Dropout is amazing and will drastically improve generalization of your model. Normally you may only want to apply dropout to fully connected layers, as shared weights in convolutional layers are good regularizers themselves. However, I did notice a slight improvement in performance when using a bit of dropout on convolutional layers, thus left it in, but kept it at minimum:

```
                Type           Size         keep_p      Dropout
 Layer 1        5x5 Conv       32           0.9         10% of neurons  
 Layer 2        5x5 Conv       64           0.8         20% of neurons
 Layer 3        5x5 Conv       128          0.7         30% of neurons
 Layer 4        FC             1024         0.5         50% of neurons
```

* **L2 Regularization**. I ended up using **lambda = 0.0001** which seemed to perform best. Important point here is that L2 loss should only include weights of the fully connected layers, and normally it doesn't include bias term. Intuition behind it being that bias term is not contributing to overfitting, as it is not adding any new degree of freedom to a model. 

* **Early stopping**. I use early stopping with a patience of **100 epochs** to capture the last best-performing weights and roll back when model starts overfitting training data. I use validation set cross entropy loss as an early stopping metric, intuition behind using it instead of accuracy is that if your model is *confident* about its predictions it should generalize better.

## Training

I have generated two datasets for training my model using augmentation pipeline I mentioned earlier:

* **Extended** dataset. This dataset simply contains **20x more data** than the original one — e.g. for each training example we generate 19 additional examples by jittering original image, with **augmentation intensity = 0.75**. 
* **Balanced** dataset. This dataset is balanced across classes and has **20.000 examples** for each class. These 20k contain original training dataset, as well as jittered images from the original training set (with **augmentation intensity = 0.75**) to complete number of examples for each class to 20.000 images.

**Disclaimer:** Training on **extended** dataset may not be the best idea, as some classes remain significantly less represented than the others there. Training a model with this dataset would make it biased towards predicting overrepresented classes. However, in our case we are trying to score highest accuracy on supplied test dataset, which (probably) follows the same classes distribution. So we are going to _cheat_ a bit and use this extended dataset for pre-training — this has proven to make test set accuracy higher (although hardly makes a model perform better "in the field"!).

I then use 25% of these augmented datasets for validation while training in 2 stages:

* **Stage 1: Pre-training**. On the first stage I pre-train the model using **extended** training dataset with TensorFlow `AdamOptimizer` and learning rate set to **0.001**. It normally stops improving after ~180 epochs, which takes ~3.5 hours on my machine equipped with Nvidia GTX1080 GPU.
* **Stage 2: Fine-tuning**. I then train the model using a **balanced** dataset with a decreased learning rate of **0.0001**.

These two training stages could easily get you past 99% accuracy on the test set. You can, however, improve model performance even further by re-generating **balanced** dataset with slightly decreased augmentation intensity and repeating 2nd fine-tuning stage a couple of times.

## Results

After a couple of fine-tuning training iterations this model scored **99.33% accuracy on the test set**, which is not too bad. As there was a total of 12,630 images that we used for testing, apparently there are **85 examples** that the model could not classify correctly.

Signs on most of those images either have artefacts like shadows or obstructing objects. There are, however, a couple of signs that were simply underrepresented in the training set — training solely on balanced datasets could potentially eliminate this issue, and using some sort of color information could definitely help as well.

In conclusion, according to different sources human performance on a similar task varies from 98.3% to 98.8%, therefore this model seems to outperform an average human. Which, I believe, is the ultimate goal of machine learning!

