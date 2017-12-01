# Digit Recognizing Project
My name is Adrian Sypos, and I am a final year student of the [B.Sc. (Hons) in Software Development](https://www.gmit.ie/software-development/bachelor-science-honours-software-development) at [GMIT](http://www.gmit.ie/).

As part of the module called [Emerging Technologies](https://emerging-technologies.github.io/), we have been asked to develop a web application in Python that will allow user to submit or draw an image containing a single digit. The application then should respond with the digit contained in the image. I will develop that project using [TensorFlow](https://www.tensorflow.org/) and [Keras Library](https://keras.io/). Wrapped into a web app using [Flask Micro Framework](http://flask.pocoo.org/).

Complete instructions to the problem can be found [here](https://emerging-technologies.github.io/problems/project.html). That project is worth 40% of the marks for this module.

Please also see the [Wiki](https://github.com/Sarlianth/digit-recognize-project/wiki) for more information.

## How to clone this repository
1. In the Clone with HTTPs section, copy the clone URL for the repository.
2. Open Git Bash.
3. Change the current working directory to the location where you want the cloned directory to be made.
4. Type `git clone`, and then paste the URL you copied in Step 2.
5. Press Enter. Your local clone will be created.

## Dependencies
#### Python
First of all you need Python, if you already have it, please skip this part and continue on.
If you're running Windows: the most stable Windows downloads are available from the [Python for Windows](https://www.python.org/downloads/windows/) page. Please download the latest release and continue on.
If you are using a Mac, see the [Python for Mac OS X](https://www.python.org/downloads/mac-osx/) page.
For other systems, or if you want to install from source, see the [general download page](https://www.python.org/downloads/).

#### Pip
If you're running Python 2.7.9+ or Python 3.4+ you should already have pip installed. If you do not, please see steps above on how to download Python - you will need it to run this solution.

#### Packages
You will need the packages listed below to be able to run this program. To install those, open your CMD and run the following command for each of them: `pip3 install [name of dependency]`
* numpy
* scipy
* Flask
* Keras
* tensorflow
* Pillow
* h5py

## How to run the application
To run the application you should follow the following steps: 
	1. Clone this repository [see above]
	2. Install all dependencies listed above (using pip3 or conda)
	3. `cd` into the root folder of the project
	4. Run the following command `python app.py`
	5. Open your favourite browser and go to the following address: `http://localhost:5000/`

## References
https://elitedatascience.com/keras-tutorial-deep-learning-in-python
https://gist.github.com/fchollet/0830affa1f7f19fd47b06d4cf89ed44d
https://www.tensorflow.org/api_docs/python/tf/get_default_graph
https://www.reddit.com/r/MachineLearning/comments/3y84hr/how_does_adam_compare_to_adadelta/
https://keras.io/backend/

