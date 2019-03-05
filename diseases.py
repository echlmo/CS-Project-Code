import cv2
import numpy as np
import math
import random


"""
Helper function for Gaussian kernel value
"""
def kernel(x):
    return round(int(x/2),-2) + 5


"""
Helper function for distorting an image with a softened mask
"""
def distortWithMask(img, distortedImg, mask):
    res_channels = []

    for c in range(0, img.shape[2]):
        a = img[:, :, c]
        b = distortedImg[:, :, c]
        m = mask[:, :, c]
        res = cv2.add(
            cv2.multiply(b, cv2.divide(np.full_like(m, 255) - m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            cv2.multiply(a, cv2.divide(m, 255.0, dtype=cv2.CV_32F), dtype=cv2.CV_32F),
            dtype=cv2.CV_8U)
        res_channels += [res]

    res = cv2.merge(res_channels)

    return res


"""
Generate glaucoma
"""
def glaucoma(img):
    # Make a copy of image to work with
    imgcop = np.copy(img)
    h, w, d = img.shape

    # Make the mask
    mask = np.zeros((h, w, d))
    cv2.circle(mask, (w // 2, h // 2), h // 4, (255, 255, 255), -1)

    # Darken image (lower saturation/brightness)
    b = 190
    M = np.ones_like(imgcop) * b
    cv2.subtract(imgcop, M, dst=imgcop)

    # Blur image and the mask
    mkern = kernel(w)
    cv2.GaussianBlur(mask, (mkern, mkern), 0, dst=mask)
    cv2.blur(imgcop, (50, 50), dst=imgcop)

    # Combine image with distort with mask applied
    final = distortWithMask(img, imgcop, mask)

    return final


"""
Generate pincushion distort
"""
def pincushion(image, strength, zoom):
    imcop = np.copy(image)
    h,w,d = image.shape

    half_w = w/2
    half_h = h/2

    rad = math.sqrt(pow(w,2) + pow(h,2))/strength

    for x in range(w):
        for y in range(h):
            new_x = x - half_w
            new_y = y - half_h

            dist = math.sqrt(pow(new_x,2) + pow(new_y,2))
            r = dist/rad

            if r == 0:
                theta = 1
            else: theta = math.atan(r) / r

            sourceX = half_w + theta * new_x * zoom
            if sourceX > w:
                sourceX = w-1
            elif sourceX < 0:
                 sourceX = 0

            sourceY = half_h + theta * new_y * zoom
            if sourceY > h:
                sourceY = h-1
            elif sourceY < 0:
                 sourceY = 0

            imcop[y][x] = image[int(sourceY)][int(sourceX)]

    return imcop


"""
Generate AMD
"""
def amd(img):
    copy = np.copy(img)

    height, width, depth = img.shape
    blot = np.ones((height, width, depth))
    mask = np.zeros_like(img)

    # Create central blob and mask
    blot = cv2.circle(blot, (width // 2, height // 2), int(height // 2), (0, 0, 0), -1)  # Create circular mask
    mask = cv2.circle(mask, (width // 2, height // 2), int(height // 2), (255, 255, 255), -1)

    # Blur the blob and mask
    kern = kernel(width * 1.25)
    k2 = kernel(height)
    blurred = cv2.GaussianBlur(blot, (kern, kern), 0)
    cv2.GaussianBlur(mask, (k2, k2), 0, dst=mask)

    copy = pincushion(copy, 8, 2)
    result = distortWithMask(copy, img, mask)

    multi = np.multiply(result / 255, blurred)
    final = (multi*255).astype(np.uint8)

    return final


"""
Helper function for spawning circles
"""
def createCircle(width, height, rad):
    w = random.randint(1, width)
    h = random.randint(1, height)
    center = [int(w), int(h)]
    radius = rad

    y, x = np.ogrid[:height, :width]
    dist_from_center = np.sqrt((x - center[0])**2 + (y-center[1])**2)

    mask = dist_from_center <= radius

    return mask

"""
Helper function for adding circles to image/mask
"""
def addCircle(image, rad):
    h,w,d = image.shape
    m = createCircle(w,h,rad)
    masked_img = image.copy()
    masked_img[m] = 0
    return masked_img


"""
Helper function to add random circles to an image
"""
def addRandCircles(img, numCircs, radiusRange):
    for i in range(numCircs):
        img = addCircle(img,random.randint(1,radiusRange))

    return img


"""
Generate DR
"""
def diabeticRet(image):
    height, width, depth = image.shape
    mask = np.ones((height, width, depth))

    mask = addRandCircles(mask,50,round((height//5),-1))
    blurred = cv2.GaussianBlur(mask, (125, 125), 0)

    # Apply pincushion for further distortion of blobs
    blurred = pincushion(blurred, 5, 2)

    multi = np.multiply(image / 255, blurred)
    result = (multi*255).astype(np.uint8)

    return result