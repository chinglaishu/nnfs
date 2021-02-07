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

fashion_mnist_labels = {
 0: 'T-shirt/top',
 1: 'Trouser',
 2: 'Pullover',
 3: 'Dress',
 4: 'Coat',
 5: 'Sandal',
 6: 'Shirt',
 7: 'Sneaker',
 8: 'Bag',
 9: 'Ankle boot'
}

image_data = cv2.imread('tshirt.png', cv2.IMREAD_GRAYSCALE)
image_data = cv2.resize(image_data, (28, 28))
image_data = 255 - image_data
image_data = (image_data.reshape(1, -1).astype(np.float32) - 127.5) / 127.5
model = Model.load('fashion_mnist.model')
confidences = model.predict(image_data)
predictions = model.output_layer_activation.predictions(confidences)
prediction = fashion_mnist_labels[predictions[0]]
print(prediction)
