import numpy as np
import os
from torch.utils.data import Dataset
from matplotlib.pylot as plt
import cv2

from augmentations import MyMirrorTransform


class InputPipeline(Dataset):
    def __init__(self, setting, data_dir, train=True):
        super().__init__()
        self.data_dir = data_dir

        self.files = os.listdir(self.data_dir)
        self.train = train

        self.mirror = MyMirrorTransform(setting)

    def __len__(self):
        return len(self.files)

    def load(self, file: str):
        '''
          Load npz files to extract image and parameters
        '''

        file = os.path.join(self.data_dir, file)
        arrays = np.load(file)
        image = arrays['image']

        x = arrays['x']
        y = arrays['y']
        r = arrays['r']
        return image, x, y, r

    def visualize(self, image: np.ndarray):
        '''
          Helper function to visualize
        '''
        fig, ax = plt.subplots()
        ax.imshow(image, cmap='gray')
        ax.set_title('Circle')
        plt.show()

    def preprocess_onlyGaussian(self, image: np.ndarray, kernel_size=(5, 5)):
        '''
        Preprocess: Remove Gaussian Noise & Normalize
        '''
        median0 = cv2.GaussianBlur(image.astype('float32'), kernel_size, 1)

        image = (median0 - np.min(median0)) / (np.max(median0) - np.min(median0))

        image = np.expand_dims(image, axis=0)
        return image

    def preprocess(self, image: np.ndarray, kernel_size=(5, 5), sigma=0.08):
        '''
        Preprocess: Remove Gaussian Noise, Non-Local means filter and Normalize
        '''
        median0 = cv2.GaussianBlur(image.astype('float32'), kernel_size, 1)
        noisy = random_noise(median0.astype('float32'), var=sigma ** 2)

        sigma_est = np.mean(estimate_sigma(noisy))
        patch_kw = dict(patch_size=5, patch_distance=6)
        image = denoise_nl_means(noisy, h=0.8 * sigma_est, sigma=sigma_est,
                                 fast_mode=False, **patch_kw)
        image = np.expand_dims(image, axis=0)
        return image

    def normalize(self, image: np.ndarray, kernel_size=(5, 5)):
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        image = np.expand_dims(image, axis=0)

        return image

    def augment(self, image, col):
        image, col = self.mirror(image, col)
        return image, col

    def __getitem__(self, idx, visualize=False):
        file = self.files[idx]

        image, x, y, r = self.load(file)

        if visualize:
            self.visualize(image)

        image = self.preprocess_onlyGaussian(image)

        image = torch.tensor(image, dtype=torch.float32)
        x = torch.tensor(x, dtype=torch.int32)
        y = torch.tensor(y, dtype=torch.int32)
        radius = torch.tensor(r, dtype=torch.int32)

        if self.train:
            image, y = self.augment(image, y)

        targets = (x, y, radius)

        targets = torch.tensor(targets)

        return image, targets
