import numpy as np
import nnfs
from nnfs.datasets import (spiral_data, vertical_data)
import matplotlib.pyplot as plt

from utils_package.layer import (Layer_Dense, Layer_Dropout)
from utils_package.activation import (Activation_ReLu,
  Activation_Softmax)
from utils_package.loss import (Loss, Loss_CategoricalCrossentropy)
from utils_package.optimizer import (Optimizer_SGD, Optimizer_Adam)
from utils_package.model import (Model)
from utils_package.accuracy import (Accuracy_Categorical)

nnfs.init()

X, y = spiral_data(samples=1000, classes=3)
X_test, y_test = spiral_data(samples=100, classes=3)

model = Model()

model.add(Layer_Dense(2, 64, weight_regularizer_L2=5e-4,
  bias_regularizer_L2=5e-4))
model.add(Activation_ReLu())
#model.add(Layer_Dropout(0.1))
model.add(Layer_Dense(64, 3))
model.add(Activation_Softmax())

model.set(
  loss=Loss_CategoricalCrossentropy(),
  optimizer=Optimizer_Adam(learning_rate=0.05, decay=5e-5),
  accuracy=Accuracy_Categorical()
)

model.finalize()

model.train(X, y, validation_data=(X_test, y_test), epochs=1000, print_every=1000)

print("end")
