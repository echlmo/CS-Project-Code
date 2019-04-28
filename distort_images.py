import argparse
import os
import shutil

from PIL import Image

# import matplotlib
# matplotlib.use("TkAgg")
# from matplotlib import pyplot as plt

from diseases import *


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


"""Distort images with glaucoma distortion and save to new folder"""
def glaucoma_distort(dir):

    # Read in and distort the images in the new folder
    ims = getAllImgs(dir)

    for i in ims:
        img = cv2.imread(i)
        distorted = glaucoma(img)
        cv2.imwrite(i,distorted)


"""Distort images with AMD distortion"""
def amd_distort(dir):

    # Read in and distort the images in the new folder
    ims = getAllImgs(dir)

    for i in ims:
        img = cv2.imread(i)
        distorted = amd(img)
        cv2.imwrite(i, distorted)

"""Distort images with DR distortion"""
def diabeticRet_distort(dir):

    # Read in and distort the images in the new folder
    ims = getAllImgs(dir)

    for i in ims:
        img = cv2.imread(i)
        distorted = diabeticRet(img)
        cv2.imwrite(i, distorted)


"""Distort all images in given directory by a particular distortion and save to a new directory.
Run using command 'python distort_images.py --img_dir <directory> --distort <distort type>'"""
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
        help='Input type of distortion: glaucoma, amd, dr'
    )
    args = parser.parse_args()

    if (args.distort).lower() == "glaucoma":
        cop_dir = shutil.copytree(args.img_dir,os.path.join(os.path.dirname(args.img_dir),"glaucoma_images"))
        glaucoma_distort(cop_dir)
    elif (args.distort).lower() == "amd":
        cop_dir = shutil.copytree(args.img_dir, os.path.join(os.path.dirname(args.img_dir), "amd_images"))
        amd_distort(cop_dir)
    elif (args.distort).lower() == "dr":
        cop_dir = shutil.copytree(args.img_dir, os.path.join(os.path.dirname(args.img_dir), "diabeticRet_images"))
        diabeticRet_distort(cop_dir)


if __name__ == '__main__':
    main()