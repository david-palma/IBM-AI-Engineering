# Deep Learning and Neural Networks with Keras

In this project is required to build a regression model using the deep learning Keras library writing Python code in a iPython Notebook. Then, it's required to experiment the model with a different number of training epochs and hidden layers with the aim of evaluating the model performance.

The final assignment is divided into 4 parts as described below.

### [Part A - Build a baseline model](DL-assignment_A.ipynb)

Use the Keras library to build a neural network with the following:

- One hidden layer of 10 nodes, and a ReLU activation function.

- Use the adam optimiser and the mean squared error as the loss function.

Then:

1. Randomly split the data into a training and test sets by holding 30% of the data for testing. You can use the train_test_split helper function from Scikit-learn.

2. Train the model on the training data using 50 epochs.

3. Evaluate the model on the test data and compute the mean squared error between the predicted concrete strength and the actual concrete strength. You can use the mean_squared_error function from Scikit-learn.

4. Repeat steps 1 - 3, 50 times, i.e., create a list of 50 mean squared errors.

5. Report the mean and the standard deviation of the mean squared errors.

### [Part B - Normalise the data](DL-assignment_B.ipynb)

Repeat Part A but use a normalised version of the data. Recall that one way to normalise the data is by subtracting the mean from the individual predictors and dividing by the standard deviation.

### [Part C - Increase the number of epochs](DL-assignment_C.ipynb)

Repeat Part B but use 100 epochs this time for training. The mean of the mean squared errors achieved using 100 epochs is almost the half of the value achieved using 50 epochs. Even though the former solution (100 epochs) is slower, it performs better than the latter solution (50 epochs).

### [Part D - Increase the number of hidden layers](DL-assignment_D.ipynb)

Repeat part B but use a neural network with the following instead:

- Three hidden layers.

- Each layer composed of 10 nodes.

- ReLU activation function.

The mean of the mean squared errors of the neural network developed in D is not much better than the network developed in B, moreover it has a higher computational time.
