# This method is reverse engineered from M.P. Rice-Fernandez et al 2019
# https://www.sciencedirect.com/science/article/pii/S0168169918301911

from methods import base
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import resample
from skimage.color import rgb2luv
import glob
from tqdm import tqdm
import multiprocessing


class RicoFernandezEtAl2019(base.BenchmarkMethod):
    def __init__(self):
        super().__init__()

        self.model = SVC(kernel='poly',
                         degree=1,
                         C=0.01,
                         random_state=42,
                         class_weight='balanced')

    def preprocess_image(self, image: np.array):

        # Do color transformations
        luv = rgb2luv(image)

        # Pad image to account for the window selection
        n_pads = 2
        luv_pad = np.stack([np.pad(luv[:, :, 0], pad_width=n_pads, mode='mean'),
                            np.pad(luv[:, :, 1], pad_width=n_pads, mode='mean'),
                            np.pad(luv[:, :, 2], pad_width=n_pads, mode='mean')], axis=2)

        sample = []

        # Extract neighbourhood pixel values for individual pixel based on 5x5 window
        for i, row in enumerate(luv):
            for j, element in enumerate(row):
                sample.append((luv_pad[i: i + 5, j: j + 5, :]).reshape(-1, 75))

        # # convert to Pixels x Channels
        sample = np.concatenate(sample, axis=0)

        # sample = luv.reshape((-1, 3))

        return sample

    def train(self, train_path='data/train', val_path='data/validation'):

        # Load all training data into memory
        x = []
        y = []
        n_jobs = 24

        for mask_path in glob.glob(train_path + '/*mask.png'):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x.append(self.preprocess_image(image_i))

        x = np.array(x)
        y = np.array(y)

        # Reshape images to individual pixels as samples
        x = x.reshape((-1, 75)).astype(np.float16)
        y = y.reshape((-1)).astype(np.uint8)

        # Total number of pixels is 1'304'625'000

        # according to the method proposed by Fernandez et al 2019 only 200 samples per image are considered
        # therefore we need to subsample the data to 192*200 = 38400
        # As we cannot pinpoint the center pixel of a plant the same way as in the original paper, we sample at random.
        x_train, y_train = resample(x, y, n_samples=38400, random_state=420)

        # Train model
        self.model.fit(x_train, y_train)

        n_images = len(glob.glob(train_path + '/*mask.png'))
        inputs = []
        for i in range(n_images):
            inputs.append(x[i*122500: (i+1)*122500])

        with multiprocessing.Pool(n_jobs) as p:
            preds = list(tqdm(p.imap(self.model.predict, inputs), total=n_images))
        preds = np.concatenate(preds)

        # Reshape back to images for appropriate metrics computations
        preds = preds.reshape((-1, 122500)).astype(np.uint8)
        y = y.reshape((-1, 122500)).astype(np.uint8)

        train_metrics = self.calculate_metrics(preds, y)

        # Load all training data into memory
        x_val = []
        y_val = []

        for mask_path in glob.glob(val_path + '/*mask.png'):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y_val.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x_val.append(self.preprocess_image(image_i))

        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # Reshape images to individual pixels as samples
        x_val = x_val.reshape((-1, 75))
        y_val = y_val.reshape((-1, 1))

        n_val_images = len(glob.glob(val_path + '/*mask.png'))
        inputs_val = []
        for i in range(n_val_images):
            inputs_val.append(x_val[i*122500: (i+1)*122500])

        with multiprocessing.Pool(n_jobs) as p:
            preds_val = list(tqdm(p.imap(self.model.predict, inputs_val), total=n_val_images))
        preds_val = np.concatenate(preds_val)

        # Reshape back to images for appropriate metrics computations
        preds_val = preds_val.reshape((-1, 122500)).astype(np.uint8)
        y_val = y_val.reshape((-1, 122500)).astype(np.uint8)

        val_metrics = self.calculate_metrics(preds_val, y_val)

        return train_metrics, val_metrics

    def test(self, test_path='data/test'):
        # Load all training data into memory
        x_test = []
        y_test = []

        for mask_path in sorted(glob.glob(test_path + '/*mask.png')):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y_test.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x_test.append(self.preprocess_image(image_i))

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Reshape images to individual pixels as samples
        x_test = x_test.reshape((-1, 75))
        y_test = y_test.reshape((-1, 1))

        n_test_images = len(glob.glob(test_path + '/*mask.png'))
        inputs_val = []
        for i in range(n_test_images):
            inputs_val.append(x_test[i * 122500: (i + 1) * 122500])

        with multiprocessing.Pool(24) as p:
            preds_test = list(tqdm(p.imap(self.model.predict, inputs_val), total=n_test_images))
        preds_test = np.concatenate(preds_test)

        # Reshape back to images for appropriate metrics computations
        preds_test = preds_test.reshape((-1, 122500)).astype(np.uint8)
        y_test = y_test.reshape((-1, 122500)).astype(np.uint8)

        test_metrics = self.calculate_metrics(preds_test, y_test)
        return test_metrics, preds_test, y_test

