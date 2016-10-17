# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Build a Traffic Sign Recognition Program

**This is a Work In Progress**

### Install

This project requires **Python 3.5** and the following Python libraries installed:

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
- Android SDK & NDK (see this [README](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/android/README.md))

If you do not have Python installed yet, it is highly recommended that you install the [Anaconda](http://continuum.io/downloads) distribution of Python, which already has the above packages and more included. Make sure that you select the Python 3.5 installer and not the Python 2.x installer. `pygame` and `OpenCV` can then be installed using one of the following commands:

Run this command at the terminal prompt to install OpenCV:

**opencv**  
`conda install -c https://conda.anaconda.org/menpo opencv3`

Run this command at the terminal prompt to install PyGame:

**PyGame:**  
Mac:  `conda install -c https://conda.anaconda.org/quasiben pygame`
Windows: `conda install -c https://conda.anaconda.org/tlatorre pygame`
Linux:  `conda install -c https://conda.anaconda.org/prkrekel pygame`

### Code

A template notebook is provided as `Traffic_Signs_Recognition.ipynb`. While no code is included in the notebook, you will be required to use the notebook to implement the basic functionality of your project and answer questions about your implementation and results. 

### Run

In a terminal or command window, navigate to the project directory that contains this README and run the following command:

```bash
jupyter notebook Traffic_Signs_Recognition.ipynb
```

This will open the Jupyter Notebook software and notebook file in your browser.


### Data

1. Download the dataset (2 options)
    - You can download the pickled dataset in which we've already resized the images to 32x32 [here](https://drive.google.com/drive/folders/0B76KYRlYCyRzYjItVFU4aV91b2c).
    - (Optional). You could also download the dataset in its original format by following the instructions [here](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). We've included the notebook we used to preprocess the data [here](./Process-Traffic-Signs.ipynb).


