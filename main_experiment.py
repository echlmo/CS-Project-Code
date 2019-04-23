import json
import pickle

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.imagenet_utils import decode_predictions
from keras_preprocessing.image import ImageDataGenerator

from diseases import *


"""
Helper function: Normalise and resize image with Glaucoma distortion
"""
def preproc_glaucoma(img):
    glauc_a = glaucoma(img)
    final = preprocess_input(glauc_a)

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


"""Main experiment function"""
def run_experiment(img_dir, distort, results_file):

    # Load the model
    model = InceptionResNetV2(include_top=True, weights="imagenet", classes=1000)

    # Get images from directory into np.array (stream with keras ImageDataGenerator)
    if (distort).lower() == "glaucoma":
        datagen = ImageDataGenerator(preprocessing_function=preproc_glaucoma)
    elif (distort).lower() == "amd":
        datagen = ImageDataGenerator(preprocessing_function=preproc_amd)
    elif (distort).lower() == "dr":
        datagen = ImageDataGenerator(preprocessing_function=preproc_diabeticRet)
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    gen = datagen.flow_from_directory(
        directory=img_dir,
        target_size=(299, 299),
        shuffle=False)
    # save_to_dir= "/" + distort + "_images/")

    filepaths = gen.filepaths

    # Run experiment, giving images to the network model from generator
    preds = model.predict_generator(gen, steps=4)
    results = decode_predictions(preds)

    # Collect and write the results out to output JSON file
    output = dict()
    for i in range(len(filepaths)):

        entry = dict()

        for j in range(len(results[i])):
            entry[j] = {"class_id": results[i][j][0],
                        "class_name": results[i][j][1],
                        "confidence": results[i][j][2].tolist()}

        output[filepaths[i]] = entry

    json.dump(output, open(results_file, "w"), indent=4, separators=(',', ': '))

    # Dump results as a pickle
    # with open('results.pickle', 'w') as outfile:
    #    pickle.dump(output, outfile)

