Installation Instructions
-------------------------

* **Step 1:** Download the project repository as a zip file from the following link: `Handwritten-Digit-and-Character-Recognition <https://github.com/Snigdha2005/Handwritten-Digit-and-Character-Recognition/tree/main>`_. After downloading, extract all files from the zip archive.

* **Step 2:** Run the interface.py file using the command - python3 interface.py

* **Step 3:** Select the option of either uploading the image drawing on canvas.

* **Step 4:** After clicking on predict, it predicts the digit or character.

* **Step 5:** You can also clear the Canvas screen.

Solution
--------

* **Step 1:** An application for handwritten character and digit recognition.
* **Step 2:** Convert handwritten text into digital format efficiently.
* **Step 3:** Use of machine learning techniques.
* **Step 4:** Dataset: English Handwritten Characters

Approach
--------

* **Methodology:**
    * **Data Preprocessing:** Load, Grayscale Conversion, resize, normalize images
    * **Model Selection:** Implement Convolutional Neural Network (CNN)
    * **Training:** Train model on diverse handwritten dataset
    * **Evaluation:** Validate accuracy and fine-tune model

Files
-----
* snigdha_handwriting directory - Images of individual characters and digits for testing
* algorithms directory - ML and CNN variation models hit and trial
* all_testing.py - Predicting each image in snigdha_handwriting directory and printing list of correct, incorrect predictions
* all_testing_100.py - Simulating each 100 times and printing number of correct, incorrect predictions for each letter and digit
* cnn.py - CNN model (main)
* data_extraction.py - extracting dataset zipfile
* data_prep.py - Flattening and reshaping training, testing data for usage as per all models
* preprocessing.py - Dataset preprocessing, feature extraction
* testing.py - Predicting one image
* image_processing.py - Black and White filter + high contrast
* image_processing_2.py - White corner mask and image adjustment
* thicken.py - Thicken letter or digit and mask borders
* interface.py - Tkinter interface for character and digit recognition

Tech Stack
----------
* **Dataset:** `English Handwritten Characters <https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset>`_. This dataset contains 3,410 images of handwritten characters in English. This is a classification dataset. It contains 62 classes with 55 images of each class. The 62 classes are 0-9, A-Z and a-z.

* **Programming Language:** Python will serve as the primary programming language for development, offering a rich ecosystem of libraries and frameworks for machine learning tasks.

* **Libraries:** TensorFlow, PyTorch, and OpenCV utilized for various aspects of the project, including deep learning model development and image processing.

* **User Interface**: Tkinter, employed for developing the user interface, providing a simple and efficient way to create GUI applications in Python for interaction with the handwritten character recognition system.

Challenges
-----------
* Noise handling (Border mask - Gaussian, white)
* Adaptive image processing implementation
* Improving accuracy
* Enabling smooth drawing on the canvas.
* Capturing the drawn strokes correctly.

Learnings
---------
* ML algorithms, evaluation metrics
* Less errors by Gen AI for ML, NN, more for UI
* Performance metrics changes when layers are added/modified
* Grouping of algorithms for improving performance metrics
* Image Processing (mainly Adaptive) Techniques
* Variations in predictions for different set of images

Future Scope
------------
* Handwritten Text Recognition
* Integration with Google Lens
* Integration with LLMs for handwritten prompts