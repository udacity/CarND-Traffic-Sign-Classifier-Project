# Content: Deep Learning
## Project: Build a Traffic Sign Recognition Program

## Project Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to create a live camera application or program that prints traffic sign names it observes in real time from images it is given. You will train that model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is properly trained, you will then test your model using a live camera application (optional) or program on newly-captured images.

## Software Requirements

This project uses the following Python libraries:

- [Juypyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)

In addition to the above, for those optionally seeking to use image processing software, you may need one of the following:
- [PyGame](http://pygame.org/)
   - Helpful links for installing PyGame:
   - [Getting Started](https://www.pygame.org/wiki/GettingStarted)
   - [PyGame Information](http://www.pygame.org/wiki/info)
   - [Google Group](https://groups.google.com/forum/#!forum/pygame-mirror-on-google-groups)
   - [PyGame subreddit](https://www.reddit.com/r/pygame/)
- [OpenCV](http://opencv.org/)

For those optionally seeking to deploy an Android application:
- Android SDK & NDK (see this [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/README.md)).

Run this command at the terminal prompt to install OpenCV:

**opencv**  
`conda install -c https://conda.anaconda.org/menpo opencv3`

Run this command at the terminal prompt to install PyGame:

**PyGame:**  
Mac:  `conda install -c https://conda.anaconda.org/quasiben pygame`
Windows: `conda install -c https://conda.anaconda.org/tlatorre pygame`
Linux:  `conda install -c https://conda.anaconda.org/prkrekel pygame`

## Starting the Project

1. Download the dataset (2 options)
    - You can download the pickled dataset in which we've already resized the images to 32x32 [here]().
    - (Optional). You could also download the dataset in its original format by following the instructions [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). We've included the notebook we used to preprocess the data [here](./Process-Traffic-Signs.ipynb).

2. Clone the project and start the notebook.

```
git clone https://github.com/domluna/traffic-signs.git
cd traffic-signs
jupyter notebook
```

3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.


## Tasks

### Project Report

You will be required to answer questions about your implementation as part of your submission in the provided `Traffic_Signs_Recognition.ipynb.` As you complete the tasks below, include thorough, detailed answers to each question *provided in italics*.

Step 1: Design and Test a Model Architecture

Design and implement a deep learning model that learns to recognize traffic signs. Train and test your model on the German Traffic Sign Dataset.

There are various aspects to consider when thinking about this problem:

- Your model can be derived from a deep feedforward net or a deep convolutional network.
- Play around preprocessing techniques (normalization, rgb to grayscale, etc)
- Number of examples per label (some have more than others).
- Generate fake data.

Here is an example of a [published baseline model on this problem](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf).

***QUESTION:*** _What approach did you take in coming up with a solution to this problem?_

***QUESTION:*** _What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.)_

***QUESTION:*** _Describe how you set up the training, validation and testing data for your model. Did you generate additional data?_

***QUESTION:*** _How did you train your model?_

### Step 2: Test a Model on Newly-Captured Images

Take several pictures of numbers that you find around you (at least five), and run them through your classifier on your computer to produce example results. Alternatively (optionally), you can try using OpenCV / SimpleCV / Pygame to capture live images from a webcam and run those through your classifier.

***QUESTION:*** _Choose five candidate images of numbers you took from around you and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult?_

***QUESTION:*** _Is your model able to perform equally well on captured pictures or a live camera stream when compared to testing on the realistic dataset?_

***QUESTION:*** _If necessary, provide documentation for how an interface was built for your model to load and classify newly-acquired images._

### Step 3: Build an Application or Program for a Model (Optional)

Take your project one step further. If you're interested, look to build an Android application or even a more robust Python program that can interface with input images and display the classified numbers and even the bounding boxes. You can for example try to build an augmented reality app by overlaying your answer on the image like the [Word Lens](https://en.wikipedia.org/wiki/Word_Lens) app does.

Loading a TensorFlow model into a camera app on Android is demonstrated in the [TensorFlow Android demo app](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android), which you can simply modify.

If you decide to explore this optional route, be sure to document your interface and implementation, along with significant results you find. You can see the additional rubric items that you could be evaluated on by [following this link](https://review.udacity.com/#!/rubrics/413/view).

## Submitting the Project

### Evaluation

Your project will be reviewed by a Udacity reviewer against the **<a href="https://review.udacity.com/#!/rubrics/413/view" target="_blank">Build a Traffic Sign Recognition Program project rubric</a>**. Be sure to review this rubric thoroughly and self-evaluate your project before submission. All criteria found in the rubric must be *meeting specifications* for you to pass.

### Submission Files

When you are ready to submit your project, collect the following files and compress them into a single archive for upload. Alternatively, upload your files to github to link to the project repository:

 - The `Traffic_Signs_Recognition.ipynb` notebook file with all questions answered and all code cells executed and displaying output.
 - An **HTML** export of the project notebook with the name **report.html**. This file *must* be present for your project to be evaluated.
 - Any additional datasets or images used for the project that are not from the German Traffic Sign Dataset.
 - For the optional image recognition software component, any additional Python files necessary to run the code.
 - For the optional Android application component, documentation for accessing the application. This should be a PDF report with the name **documentation.pdf**

Once you have collected these files and reviewed the project rubric, proceed to the project submission page.

### I'm Ready!

When you're ready to submit your project, click on the **Submit Project** button at the bottom of the page.

If you are having any problems submitting your project or wish to check on the status of your submission, please email us at **machine-support@udacity.com** or visit us in the <a href="http://discussions.udacity.com" target="_blank">discussion forums</a>.

### What's Next?

You will get an email as soon as your reviewer has feedback for you. In the meantime, review your next project and feel free to get started on it or the courses supporting it!
