import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score, accuracy_score, confusion_matrix
import PIL

pd.options.display.max_columns = None


class BenchmarkMethod:
    # Implement the following methods
    def train(self, train_path='../data/train', val_path='../data/val'):
        return None, None

    def test(self, test_path='data/test'):
        return None

    def preprocess_image(self, image: np.array):
        pass

    @staticmethod
    def calculate_metrics(preds: np.array, y: np.array):

        classes_labels = {'plants': 0, 'soil': 1}

        score_names = ['f1', 'precision', 'recall', 'IoU']
        columns = [name + ' - class: ' + class_label for name in score_names for class_label in
                   classes_labels.keys()]
        columns.append('accuracy: ')
        statistics = pd.DataFrame(columns=columns)

        # iterate sample wise through the data and average the metrics wrt. to individual images
        for pred_i, y_i in zip(preds, y):
                scores = {}

                for class_label in classes_labels.keys():
                    i = classes_labels[class_label]
                    scores.update({('f1 - class: ' + class_label): f1_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)})
                    scores.update({('precision - class: ' + class_label): precision_score(y_i, pred_i, average='binary',
                                                                                          pos_label=i, zero_division=0)})
                    scores.update(
                        {('recall - class: ' + class_label): recall_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)})
                    scores.update(
                        {('IoU - class: ' + class_label): jaccard_score(y_i, pred_i, average='binary', pos_label=i, zero_division=0)})

                scores.update({'accuracy: ': accuracy_score(y_i, pred_i)})

                statistics = statistics.append(scores, ignore_index=True)

        return statistics.mean(axis=0).to_dict()

    @staticmethod
    def read_image(imagepath, rgb: bool, metadata: bool = False):
        # read image in a standardized manner
        image = PIL.Image.open(imagepath)

        if metadata:
            metadata = image.text

        if rgb:
            pass
        else:
            image = image.convert('L')

        image = np.array(image)

        # normalize to [0,1]
        if image.max() > 1:  # Assume uint8 format
            image = image / 255.0

        if not rgb:
            # Encode the greyscale masks
            image = np.where(image > 0.5, np.ones_like(image), np.zeros_like(image))

        if metadata:
            return image, metadata

        return image
