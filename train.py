import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils_package.layer import (Layer_Dense, Layer_Dropout)
from utils_package.activation import (Activation_ReLu,
  Activation_Softmax)
from utils_package.loss import (Loss, Loss_CategoricalCrossentropy)
from utils_package.optimizer import (Optimizer_SGD, Optimizer_Adam)
from utils_package.model import (Model)
from utils_package.accuracy import (Accuracy_Categorical)

def load_mnist_dataset(dataset, path):
  labels = os.listdir(os.path.join(path, dataset))
  print(labels)
  X = []
  y = []
  index = 0
  for label in labels:
    for file in os.listdir(os.path.join(path, dataset, label)):
      image = cv2.imread(os.path.join(path, dataset, label, file),
        cv2.IMREAD_UNCHANGED)
      X.append(image)
      y.append(label)
      index += 1
      if index > 100:
        break
  return np.array(X), np.array(y).astype('uint8')

def create_data_mnist(path):
  X, y = load_mnist_dataset("train", path)
  X_test, y_test = load_mnist_dataset("test", path)
  return X, y, X_test, y_test

X, y, X_test, y_test = create_data_mnist("fashion_mnist_images")

keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)

X = X[keys]
y = y[keys]

X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

model = Model()
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLu())
model.add(Layer_Dense(128, 128))
model.add(Activation_ReLu())
model.add(Layer_Dense(128, 10)) # output
model.add(Activation_Softmax())

model.set(loss=Loss_CategoricalCrossentropy(),
          optimizer=Optimizer_Adam(decay=1e-3),
          accuracy=Accuracy_Categorical())

model.finalize()

# model.load_parameters("fashion_mnist.params")

model.evaluate(X_test, y_test)

model.train(X, y, validation_data=(X_test, y_test),
  epochs=10, batch_size=128, print_every=100)

model.save_parameters('fashion_mnist.parms')
model.save("fashion_mnist.model")
