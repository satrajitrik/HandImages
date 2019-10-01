import cv2
import numpy as np

from skimage.feature import local_binary_pattern


class Descriptor(object):
    def __init__(self, image):
        self.image = image
    
    def sift(self):
        orb = cv2.ORB_create()
        sift = cv2.xfeatures2d.SIFT_create()
        grey_scale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        kp, des = sift.detectAndCompute(grey_scale_image, None)
        
        return des
    
    def lbp(self):
        radius = 1
        n_points = 8 * radius
        
        grey_scale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        blocks = np.array(
            [grey_scale_image[x:x + 100, y:y + 100] for x in range(0, grey_scale_image.shape[0], 100) for y in
             range(0, grey_scale_image.shape[1], 100)])
        lbps = np.array([local_binary_pattern(block, n_points, radius, 'default').reshape(10000, ) for block in blocks])
        lbp_histograms = np.array([np.histogram(lbp, bins=np.arange(257), density=True)[0] for lbp in lbps])
        
        concat_histograms = lbp_histograms[0]
        
        for i in range(1, len(lbp_histograms)):
            concat_histograms = np.concatenate([concat_histograms, lbp_histograms[i]])
        
        return concat_histograms
    
    def hog():
        pass
    
    def color_moments():
        pass
