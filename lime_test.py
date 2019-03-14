import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from lime import lime_image
from skimage.segmentation import mark_boundaries
from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from diseases import glaucoma, diabeticRet, amd


model = InceptionResNetV2(include_top=True,weights="imagenet",classes=1000)

explainer = lime_image.LimeImageExplainer()

# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input)
#
# gen = datagen.flow_from_directory(
#     directory="/Users/echlmo/Desktop/testing_images/",
#     target_size=(299,299),
#     shuffle=False)
#
# filepaths = gen.filepaths
#
# images = []
# x,_ = gen.next()
# for i in range(1):
#     images.append(x[i])
# imgs = np.vstack(images)
#
# out = []

# preds = model.predict(images)
# preds = model.predict_generator(gen, steps=4)
# decoded = decode_predictions(preds)

dog = cv2.imread("./dog.jpg")[...,::-1]
resized = np.array([cv2.resize(dog, (299, 299))])
images = preprocess_input(resized)

explanation = explainer.explain_instance(
    images[0],
    model.predict,
    top_labels=5, hide_color=0)

temp, mask = explanation.get_image_and_mask(208, positive_only=True, num_features=5, hide_rest=True)
plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
plt.show()

