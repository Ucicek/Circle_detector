import numpy as np
import torchvision.transforms.functional as TF

class MyMirrorTransform:
    """Rotate by one of the given angles."""
    def __init__(self, setting):
        self.setting = setting
        self.parameters = self.setting.Mirror
        self.apply_prob = np.random.rand(1)

    def apply_image(self, image):
        return TF.hflip(image)

    def apply_keypoints(self,image,col):
      mid_point = int(image.shape[1]/2)
      if col > mid_point:
          distance = col - mid_point
          col = mid_point - distance
      else:
          distance = mid_point - col
          col = mid_point + distance
      return col

    def __call__(self, image, col):
        if self.parameters['prob'] > self.apply_prob:
          return self.apply_image(image), self.apply_keypoints(image,col)
        return image,col