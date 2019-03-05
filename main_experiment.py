import argparse

import pickle
import json

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



"""
Main function: Run custom experiment from console as a python command
"""
def main():

    # Set up command and arguments to input
    parser = argparse.ArgumentParser(description='Run experiment')
    parser.add_argument(
        '--img_dir',
        default='/',
        type=str,
        help='Input path to directory of images to test.'
    )
    parser.add_argument(
        '--distort',
        default='none',
        type=str,
        help='Input type of distortion: none (default), glaucoma, amd, dr'
    )
    args = parser.parse_args()

    # Load the model
    model = InceptionResNetV2(include_top=True,weights="imagenet",classes=1000)

    # Get images from directory into np.array (stream with keras ImageDataGenerator)
    if (args.distort).lower() == "glaucoma":
        datagen = ImageDataGenerator(preprocessing_function=preproc_glaucoma)
    elif (args.distort).lower() == "amd":
        datagen = ImageDataGenerator(preprocessing_function=preproc_amd)
    elif (args.distort).lower() == "dr":
        datagen = ImageDataGenerator(preprocessing_function=preproc_diabeticRet)
    else:
        datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    gen = datagen.flow_from_directory(
        directory=args.img_dir,
        target_size=(299, 299),
        shuffle=False)
        # save_to_dir= "/" + args.distort + "_images/")

    filepaths = gen.filepaths

    # Run experiment, giving images to the network model from generator
    step = int(math.ceil(len(gen)/32))
    preds = model.predict_generator(gen, steps=step)
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

    json.dump(output, open("./results.json", "w"), indent=4, separators=(',', ': '))

    # Dump results as a pickle
    # out = dict(zip(filepaths,results))
    # with open('results.pickle', 'w') as outfile:
    #    pickle.dump(out, outfile)


if __name__ == '__main__':
    main()