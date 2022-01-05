# This method is reverse engineered from Kang Yu et al. 2017
# https://link.springer.com/article/10.1186/s13007-017-0168-4

from methods import base
import numpy as np
from sklearn.svm import SVC
from sklearn.utils import resample
from skimage.color import rgb2hsv, rgb2lab
import glob
from os.path import split
import time
from tqdm import tqdm
import multiprocessing


class YuEtAl2017(base.BenchmarkMethod):
    def __init__(self):
        super().__init__()

        # Pure SVC
        self.ilum_model = SVC(kernel='linear',
                              C=0.01,
                              random_state=42,
                              class_weight='balanced')

        self.pixel_model_llc = SVC(kernel='rbf',
                                   C=0.01,
                                   random_state=42)

        self.pixel_model_hlc = SVC(kernel='rbf',
                                   C=0.01,
                                   random_state=42)

    def preprocess_image(self, image: np.array):

        histograms = np.concatenate((np.histogram(image[:, :, 0], bins=256, range=(0, 1))[0],
                                     np.histogram(image[:, :, 1], bins=256, range=(0, 1))[0],
                                     np.histogram(image[:, :, 2], bins=256, range=(0, 1))[0]))

        hsv = rgb2hsv(image)
        lab = rgb2lab(image)
        ndi3 = (image[:, :, 0] + image[:, :, 1] - 2 * image[:, :, 2]) / \
               (image[:, :, 0] + image[:, :, 1] + 2 * image[:, :, 2])

        ndi3[np.isnan(ndi3)] = 0
        ndi3[np.isinf(ndi3)] = 0

        sample = np.concatenate((image, hsv, lab, np.multiply(ndi3, hsv[:, :, 2]).reshape((*ndi3.shape, 1))), axis=2)

        # reshape to  pixels x channels
        sample = sample.reshape((-1, 10))

        return sample, histograms

    def train(self, train_path='data/Data for Yu et al/train', val_path='data/Data for Yu et al/validation'):

        # Load all training data into memory
        names = []
        x_img_llc = []
        y_img_llc = []
        x_img_hlc = []
        y_img_hlc = []
        x_hist = []
        y_hist = []
        x_all = []
        y_all = []
        n_threads = 24

        types = ['HLC', 'LLC']
        for type in types:
            for mask_path in glob.glob(train_path + '/' + type + '/' + '/*mask.png'):
                image_path = mask_path.replace('_mask', '')
                _, name = split(image_path)
                names.append(name)

                mask_i = self.read_image(mask_path, rgb=False)
                image_i = self.read_image(image_path, rgb=True)
                sample, hist = self.preprocess_image(image_i)
                x_hist.append(hist)

                if type == 'HLC':
                    y_hist.append(0)
                    x_img_hlc.append(sample)
                    y_img_hlc.append(mask_i.reshape((-1, 1)))
                elif type == 'LLC':
                    y_hist.append(1)
                    x_img_llc.append(sample)
                    y_img_llc.append(mask_i.reshape((-1, 1)))
                else:
                    raise Exception("Unknown type")

                x_all.append(sample)
                y_all.append(mask_i.reshape((-1, 1)))

        x_all = np.array(x_all)
        y_all = np.array(y_all)

        x_img_llc = np.array(x_img_llc).reshape((-1, 10))
        x_img_hlc = np.array(x_img_hlc).reshape((-1, 10))
        y_img_llc = np.array(y_img_llc).reshape((-1))
        y_img_hlc = np.array(y_img_hlc).reshape((-1))

        # Reshape images to individual pixels as samples
        x_hist = np.array(x_hist).reshape((-1, 256*3))
        y_hist = np.array(y_hist).reshape((-1))
        # Training time of 350s with 122500 samples, 780s with 2*122500 samples, 6019s with 4*122500 samples, 8*122500 did not converge
        n_samples_llc = 4*122500 if 4*122500 < y_img_llc.shape[0] else y_img_llc.shape[0]
        n_samples_hlc = 4*122500 if 4*122500 < y_img_hlc.shape[0] else y_img_hlc.shape[0]

        x_img_llc_train, y_img_llc_train = resample(x_img_llc, y_img_llc, n_samples=n_samples_llc, random_state=420)
        x_img_hlc_train, y_img_hlc_train = resample(x_img_hlc, y_img_hlc, n_samples=n_samples_hlc, random_state=420)

        # Train model
        t = time.time()
        self.pixel_model_llc.fit(x_img_llc_train, y_img_llc_train)
        self.pixel_model_hlc.fit(x_img_hlc_train, y_img_hlc_train)
        self.ilum_model.fit(x_hist, y_hist)

        n_images = 0
        types = ['HLC', 'LLC']
        for type in types:
            n_images += len(glob.glob(train_path + '/' + type + '/' + '/*mask.png'))

        hists = []
        for i in range(n_images):
            hists.append(x_hist[i].reshape((1, -1)))

        with multiprocessing.Pool(n_threads) as p:
            ilum_preds = list(tqdm(p.imap(self.ilum_model.predict, hists), total=n_images))
        ilum_preds = np.concatenate(ilum_preds)

        llc_inputs = x_all[ilum_preds == 1]
        hlc_inputs = x_all[ilum_preds == 0]

        with multiprocessing.Pool(n_threads) as p:
            llc_preds = list(tqdm(p.imap(self.pixel_model_llc.predict, llc_inputs), total=llc_inputs.shape[0]))
        llc_preds = np.concatenate(llc_preds)
        with multiprocessing.Pool(n_threads) as p:
            hlc_preds = list(tqdm(p.imap(self.pixel_model_hlc.predict, hlc_inputs), total=hlc_inputs.shape[0]))
        hlc_preds = np.concatenate(hlc_preds)

        preds = np.zeros(x_all.shape[0:2])

        # Reshape back to images for appropriate metrics computations
        hlc_preds = hlc_preds.reshape((-1, 122500))
        llc_preds = llc_preds.reshape((-1, 122500))

        preds[ilum_preds == 0] = hlc_preds
        preds[ilum_preds == 1] = llc_preds

        # Reshape back to images for appropriate metrics computations
        preds = preds.reshape((-1, 122500))

        train_metrics = self.calculate_metrics(preds, y_all)

        names = []
        x_val_hist = []
        y_val_hist = []
        x_val = []
        y_val = []
        types = ['HLC', 'LLC']
        for type in types:
            for mask_path in glob.glob(val_path + '/' + type + '/' + '/*mask.png'):
                image_path = mask_path.replace('_mask', '')
                _, name = split(image_path)
                names.append(name)

                mask_i = self.read_image(mask_path, rgb=False)
                image_i = self.read_image(image_path, rgb=True)
                sample, hist = self.preprocess_image(image_i)
                x_val_hist.append(hist)

                x_val.append(sample)
                y_val.append(mask_i.reshape((-1, 1)))

                if type == 'HLC':
                    y_val_hist.append(0)
                elif type == 'LLC':
                    y_val_hist.append(1)
                else:
                    raise Exception("Unknown type")

        x_val = np.array(x_val)
        y_val = np.array(y_val)

        # Reshape images to individual pixels as samples
        x_val_hist = np.array(x_val_hist).reshape((-1, 256 * 3))
        y_val_hist = np.array(y_val_hist).reshape((-1))

        print(self.ilum_model.score(x_val_hist, y_val_hist))

        n_val_images = 0
        types = ['HLC', 'LLC']
        for type in types:
            n_val_images += len(glob.glob(val_path + '/' + type + '/' + '/*mask.png'))

        hists = []
        for i in range(n_val_images):
            hists.append(x_val_hist[i].reshape((1, -1)))

        with multiprocessing.Pool(n_threads) as p:
            ilum_preds = list(tqdm(p.imap(self.ilum_model.predict, hists), total=n_val_images))
        ilum_preds = np.concatenate(ilum_preds)

        llc_inputs = x_val[ilum_preds == 1]
        hlc_inputs = x_val[ilum_preds == 0]

        with multiprocessing.Pool(n_threads) as p:
            llc_preds = list(tqdm(p.imap(self.pixel_model_llc.predict, llc_inputs), total=llc_inputs.shape[0]))
        llc_preds = np.concatenate(llc_preds)
        with multiprocessing.Pool(n_threads) as p:
            hlc_preds = list(tqdm(p.imap(self.pixel_model_hlc.predict, hlc_inputs), total=hlc_inputs.shape[0]))
        hlc_preds = np.concatenate(hlc_preds)

        preds = np.zeros(x_val.shape[0:2])

        # Reshape back to images for appropriate metrics computations
        hlc_preds = hlc_preds.reshape((-1, 122500))
        llc_preds = llc_preds.reshape((-1, 122500))

        preds[ilum_preds == 0] = hlc_preds
        preds[ilum_preds == 1] = llc_preds

        val_metrics = self.calculate_metrics(preds, y_val)

        return train_metrics, val_metrics

    def test(self, test_path='data/Data for Yu et al/test'):
        n_threads = 24
        names = []
        x_test_hist = []
        y_test_hist = []
        x_test = []
        y_test = []
        types = ['HLC', 'LLC']
        for type in types:
            for mask_path in sorted(glob.glob(test_path + '/' + type + '/' + '/*mask.png')):
                image_path = mask_path.replace('_mask', '')
                _, name = split(image_path)
                names.append(name)

                mask_i = self.read_image(mask_path, rgb=False)
                image_i = self.read_image(image_path, rgb=True)
                sample, hist = self.preprocess_image(image_i)
                x_test_hist.append(hist)

                x_test.append(sample)
                y_test.append(mask_i.reshape((-1, 1)))

                if type == 'HLC':
                    y_test_hist.append(0)
                elif type == 'LLC':
                    y_test_hist.append(1)
                else:
                    raise Exception("Unknown type")

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # Reshape images to individual pixels as samples
        x_test_hist = np.array(x_test_hist).reshape((-1, 256 * 3))
        y_test_hist = np.array(y_test_hist).reshape((-1))

        print(self.ilum_model.score(x_test_hist, y_test_hist))

        n_test_images = 0
        types = ['HLC', 'LLC']
        for type in types:
            n_test_images += len(glob.glob(test_path + '/' + type + '/' + '/*mask.png'))

        hists = []
        for i in range(n_test_images):
            hists.append(x_test_hist[i].reshape((1, -1)))

        with multiprocessing.Pool(n_threads) as p:
            ilum_preds = list(tqdm(p.imap(self.ilum_model.predict, hists), total=n_test_images))
        ilum_preds = np.concatenate(ilum_preds)

        llc_inputs = x_test[ilum_preds == 1]
        hlc_inputs = x_test[ilum_preds == 0]

        with multiprocessing.Pool(n_threads) as p:
            llc_preds = list(tqdm(p.imap(self.pixel_model_llc.predict, llc_inputs), total=llc_inputs.shape[0]))
        llc_preds = np.concatenate(llc_preds)
        with multiprocessing.Pool(n_threads) as p:
            hlc_preds = list(tqdm(p.imap(self.pixel_model_hlc.predict, hlc_inputs), total=hlc_inputs.shape[0]))
        hlc_preds = np.concatenate(hlc_preds)

        preds = np.zeros(x_test.shape[0:2])

        # Reshape back to images for appropriate metrics computations
        hlc_preds = hlc_preds.reshape((-1, 122500))
        llc_preds = llc_preds.reshape((-1, 122500))

        preds[ilum_preds == 0] = hlc_preds
        preds[ilum_preds == 1] = llc_preds

        # Reshape back to images for appropriate metrics computations
        preds = preds.reshape((-1, 122500))

        test_metrics = self.calculate_metrics(preds, y_test)
        return test_metrics, preds, y_test
