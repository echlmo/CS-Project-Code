import os
import numpy as np
import cv2
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

from PIL import Image

from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
import keras.backend as K


"""Get all (image) files in directory tree"""
def getAllImgs(dir):

    fileList = []
    for (dirpath, dirnames, filenames) in os.walk(dir):
        fileList += [os.path.join(dirpath, file) for file in filenames]

    for f in fileList:
        try:
            img = Image.open(f)
            img.verify()
        except OSError as e:
            fileList.remove(f)
        except IOError as e:
            fileList.remove(f)

    return fileList


"""Preprocess image"""
def preproc(img):
    im = cv2.imread(img)[...,::-1]
    resized = cv2.resize(im, (299, 299))  # Resize to InceptionResNet 299x299 sizing
    expanded = np.expand_dims(resized,axis=0)
    final = preprocess_input(expanded)
    return final


"""Get the gradient class activation mapping for a single image"""
def get_gradcam_single(image, model, dest):

    x = preproc(image)

    preds = model.predict(x)
    class_idx = np.argmax(preds[0])
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer('conv_7b')

    grads = K.gradients(class_output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(1536):
        conv_layer_output_value[:, :, i] *= pooled_grads_value[i]

    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)

    original = cv2.imread(image)[..., ::-1]
    heatmap = cv2.resize(heatmap, (original.shape[1], original.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    cv2.imwrite(dest + os.path.basename(image), superimposed_img)


"""Get GradCAM for all images in a directory"""
def get_gradcam_images(images_path, model, dest):

    if not os.path.exists(dest):
        os.makedirs(dest)

    imlist = getAllImgs(images_path)

    for x in imlist:
        get_gradcam_single(x, model, dest)
