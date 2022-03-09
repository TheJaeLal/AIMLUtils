import cv2
import numpy as np

class ImageFeatureExtractor:
    
    def __init__(self, _type='SVD', num_features = 200):
        
        self.num_features = num_features
        self._type = _type
        
        if _type == 'ORB':
            self._extractor = cv2.ORB_create(num_features)
        elif _type == 'SVD':
            self._extractor = np.linalg.svd
        else:
            raise NotImplementedError("Only ORB features supported as of now")
            
        return
    
    def extract(self, img):
        
        if self._type == 'SVD':
            dsc = self._extractor(img, compute_uv=False)
            
        else:
            # Dinding image keypoints
            kps = self._extractor.detect(img)

            # Getting first 32 of them. 
            # Number of keypoints is varies depend on image size and color pallet
            # Sorting them based on keypoint response value(bigger is better)
            kps = sorted(kps, key=lambda x: -x.response)[:self.num_features]

            # computing descriptors vector
            kps, dsc = self._extractor.compute(img, kps)

        if dsc is None:
            return dsc
        
        return dsc.flatten()
    