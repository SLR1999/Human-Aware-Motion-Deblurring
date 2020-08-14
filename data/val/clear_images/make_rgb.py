import cv2
import numpy as np
from PIL import Image

im_cv = cv2.imread('src_1.png')

cv2.imwrite('src_rgb.png', im_cv)