import os
import sys
import cv2 as cv
import matplotlib.pyplot as plt

def load_img(img, mode='RGB'):
    base_path = './../image'
    if mode == 'RGB':
        img = cv.imread(os.path.join(base_path, img))
    else:
        img = cv.imread(os.path.join(base_path, img), cv.IMREAD_GRAYSCALE)

    if img is None:
        print('Image load failed!')
        sys.exit()
    return img
    
def show_imgs(imgs, mode='RGB'):
    plt.figure(figsize=(12, 10))
    t_len = len(imgs)
    if mode == 'RGB':
        for i, img in enumerate(imgs):
            plt.subplot(1, t_len, i+1)
            plt.imshow(img)
            plt.xticks([])
            plt.yticks([])
    else:
        for i, img in enumerate(imgs):
            plt.subplot(1, t_len, i+1)
            plt.imshow(img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
    
    plt.show()
