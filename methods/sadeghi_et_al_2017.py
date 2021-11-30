# This Method is reverse engineered from Pouria Sadeghi-Tehran et al 2017
# https://plantmethods.biomedcentral.com/articles/10.1186/s13007-017-0253-8

from methods import base
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from skimage.color import rgb2lab, rgb2luv, rgb2hsv, rgb2ycbcr, rgb2yuv
import glob


class SadeghiEtAl2017(base.BenchmarkMethod):
    def __init__(self):
        super().__init__()
        self.scaler = StandardScaler()

        self.model = RandomForestClassifier(max_depth=95,
                                            max_features=6,
                                            min_samples_leaf=6,
                                            min_samples_split=4,
                                            n_estimators=55,
                                            bootstrap=False,
                                            random_state=42,
                                            n_jobs=-1)

    def preprocess_image(self, image: np.array):
        # Reshape to RGB vector
        rgb = image.reshape((-1, image.shape[2]))

        # Do color transformations
        lab = rgb2lab(rgb)
        luv = rgb2luv(rgb)
        hsv = rgb2hsv(rgb)
        hsi = rgb2hsi(rgb)
        ycbcr = rgb2ycbcr(rgb)
        yuv = rgb2yuv(rgb)

        # merge to single array
        sample = np.concatenate((rgb, lab, luv, hsv, hsi, ycbcr, yuv), axis=1)

        # Convert to Channels x Pixels
        sample = sample.reshape((21, -1))

        return sample

    def train(self, train_path='data/train', val_path='data/validation'):

        # Load all training data into memory
        x = []
        y = []

        for mask_path in glob.glob(train_path + '/*mask.png'):
            image_path = mask_path.replace('_mask', '')

            mask_i = self.read_image(mask_path, rgb=False)
            y.append(mask_i.reshape((-1, 1)))

            image_i = self.read_image(image_path, rgb=True)
            x.append(self.preprocess_image(image_i))

        x = np.array(x)
        y = np.array(y)

        # Reshape images to individual pixels as samples
        x = x.reshape((-1, 21))
        y = y.reshape((-1,)).astype(np.uint8)

        x = self.scaler.fit_transform(x)

        # Train model
        self.model.fit(X=x, y=y)

        # Calculate training metrics
        preds = self.model.predict(x)

        # Reshape back to images for appropriate metrics computations
        preds = preds.reshape((-1, 122500)).astype(np.uint8)
        y = y.reshape((-1, 122500))

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
        x_val = x_val.reshape((-1, 21))
        y_val = y_val.reshape((-1,)).astype(np.uint8)

        x_val = self.scaler.transform(x_val)

        # Calculate training metrics
        preds_val = self.model.predict(x_val)

        # Reshape back to images for appropriate metrics computations
        preds_val = preds_val.reshape((-1, 122500))
        y_val = y_val.reshape((-1, 122500))

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
        x_test = x_test.reshape((-1, 21))
        y_test = y_test.reshape((-1, 1))

        x_test = self.scaler.transform(x_test)

        # Calculate training metrics
        preds_test = self.model.predict(x_test)

        # Reshape back to images for appropriate metrics computations
        preds_test = preds_test.reshape((-1, 122500)).astype(np.uint8)
        y_test = y_test.reshape((-1, 122500)).astype(np.uint8)

        test_metrics = self.calculate_metrics(preds_test, y_test)
        return test_metrics, preds_test, y_test


def rgb2hsi(rgb: np.array):
    # Method according to https://www.vocal.com/video/rgb-and-hsvhsihsl-color-space-conversion/

    h = rgb2hsv(rgb)[:, 0]
    c_min = np.min(rgb, axis=1)
    i = np.mean(rgb, axis=1)
    s = np.where(i != 0, 1 - c_min / i, np.zeros_like(i))

    return np.stack((h, s, i), axis=1)
