# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Initialize Parameters
2. Compute Prediction (Forward Pass)
3. Compute Gradients
4. Update Parameters

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: vardhansai I 
RegisterNumber:  25015695
*/
import numpy as np
import matplotlib.pyplot as plt

# Sample dataset
X = np.array([0,1,2,3,4,5,6,7,8,9])
Y = np.array([0,0,0,0,0,1,1,1,1,1])

# Initialize parameters
w = 0
b = 0

learning_rate = 0.01
epochs = 1000
n = len(X)

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent
for i in range(epochs):

    # Linear model
    z = w * X + b

    # Prediction
    y_pred = sigmoid(z)

    # Gradients
    dw = (1/n) * np.sum((y_pred - Y) * X)
    db = (1/n) * np.sum(y_pred - Y)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

print("Weight:", w)
print("Bias:", b)

# Predictions
z = w * X + b
prob = sigmoid(z)

plt.scatter(X, Y, color="blue", label="Actual Data")
plt.plot(X, prob, color="red", label="Logistic Curve")
plt.xlabel("X")
plt.ylabel("Probability")
plt.legend()
plt.show()
```

## Output:
![logistic regression using gradient descent](sam.png)
X	Probability
0	~0.003
3	~0.10
5	~0.50
7	~0.90
9	~0.99

## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

