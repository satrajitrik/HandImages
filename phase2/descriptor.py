import cv2
import numpy as np
import skimage.feature as sk_feature
import skimage.transform as sk_transform
from skimage.feature import local_binary_pattern


class Descriptor(object):
    def __init__(self, image):
        self.image = image
    
    def sift(self):
        orb = cv2.ORB_create()
        sift = cv2.xfeatures2d.SIFT_create()
        grey_scale_image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptor = sift.detectAndCompute(grey_scale_image, None)
        return descriptor
    
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
    
    def hog(self):
        scaled_image = sk_transform.rescale(self.image, 0.1,
                                            anti_aliasing=True)  # Anti-aliasing applies gaussian filter
        hog_feature_vector, hog_image = sk_feature.hog(scaled_image, orientations=9, pixels_per_cell=(8, 8),
                                                       cells_per_block=(2, 2), block_norm='L2-Hys',
                                                       visualize=True, feature_vector=True, multichannel=True)
        return hog_feature_vector
    
    def color_moments(self):
        pass
