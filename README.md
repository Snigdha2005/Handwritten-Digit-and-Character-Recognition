# Handwritten-Digit-and-Character-Recognition

In today’s digitized world, there is a vast amount of handwritten data that requires efficient processing and analysis. Handwritten digit and character recognition systems play a crucial role in converting handwritten text into digital format, enabling automation in various industries such as finance, postal services, and document management. This project proposal aims to develop a robust and accurate handwritten digit and character recognition system using machine learning techniques.

## TechStack

- Programming Language - Python
- Libraries - Tensorflow, PyTorch, OpenCV
- User Interface - Tkinter

## Files

- data directory - https://www.kaggle.com/datasets/dhruvildave/english-handwritten-characters-dataset?resource=download
- snigdha_handwriting directory - Images of individual characters and digits for testing
- algorithms directory - ML and CNN variation models hit and trial
- all_testing.py - Predicting each image in snigdha_handwriting directory and printing list of correct, incorrect predictions
- all_testing_100.py - Simulating each 100 times and printing number of correct, incorrect predictions for each letter and digit
- cnn.py - CNN model (main) 
- data_extraction.py - extracting dataset zipfile
- data_prep.py - Flattening and reshaping training, testing data for usage as per all models
- preprocessing.py - Dataset preprocessing, feature extraction
- testing.py - Predicting one image