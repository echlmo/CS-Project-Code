import cv2
import json
import numpy as np

from keras_preprocessing.image import ImageDataGenerator
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from diseases import glaucoma, diabeticRet, amd

from lime import lime_image


"""
Helper function: Normalise and resize image with Glaucoma distortion
"""
def preproc_glaucoma(img):
    glauc_a = glaucoma(img)
    resized = np.array([cv2.resize(glauc_a, (299, 299))])
    final = preprocess_input(resized)

    return final


"""
Helper function: Normalise and resize image with AMD distortion
"""
def preproc_amd(img):
    amd_a = amd(img)
    final = preprocess_input(amd_a)

    return final


"""
Helper function: Normalise and resize image with DR distortion
"""
def preproc_diabeticRet(img):
    dr_a = diabeticRet(img)
    final = preprocess_input(dr_a)

    return final


model = InceptionResNetV2(include_top=True,weights="imagenet",classes=1000)

explainer = lime_image.LimeImageExplainer()

# datagen = ImageDataGenerator(
#     preprocessing_function=preprocess_input)
# datagen = ImageDataGenerator(
#      preprocessing_function=preproc_glaucoma)
datagen = ImageDataGenerator(
     preprocessing_function=preproc_amd)
# datagen = ImageDataGenerator(
#      preprocessing_function=preproc_diabeticRet)

gen = datagen.flow_from_directory(
    directory="/Users/echlmo/Desktop/testing_images/",
    # directory="/Users/echlmo/Desktop/amd_images/",
    target_size=(299,299),
    shuffle=False)

filepaths = gen.filepaths

preds = model.predict_generator(gen, steps=4)
decoded = decode_predictions(preds)

explanation = explainer.explain_instance(
    images[0],
    preds,
    top_labels=5, hide_color=0)

poo = dict()
for i in range(len(filepaths)):

    entry = dict()

    for j in range(len(decoded[i])):
        entry[j] = {"class_id": decoded[i][j][0],
                    "class_name": decoded[i][j][1],
                    "confidence": decoded[i][j][2].tolist()}

    poo[filepaths[i]] = entry


# json.dump(poo,open("./results.json","w"),indent=4,separators=(',',': '))
# json.dump(poo,open("./glaucoma_results.json","w"),indent=4,separators=(',',': '))
# json.dump(poo,open("./amd_results-savetest.json","w"),indent=4,separators=(',',': '))
# json.dump(poo,open("./diabeticRet_results.json","w"),indent=4,separators=(',',': '))