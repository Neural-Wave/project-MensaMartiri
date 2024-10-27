import config
import utils
from patch_classifier import PatchClassifier
from gaussian_point_finder import finder_class
from PIL import Image
from torchmetrics.classification import BinaryAccuracy, FBetaScore, Precision, Recall
from torchvision import transforms
import torch


class Predictor:

    def __init__(self, model_path):
        self.detector = finder_class()
        self.classifier = PatchClassifier.load_from_checkpoint(model_path)
        self.accuracy = BinaryAccuracy(threshold=0.5)
        self.fb = FBetaScore(task="binary", beta=0.5, threshold=0.5)
        self.precision = Precision(task="binary", num_classes=2)
        self.recall = Recall(task="binary", num_classes=2)
        self.transform = transforms.ToTensor()

    def predict(self, tensor):

        coordinates = self.detector.point_finder(tensor)

        # No corners detected, assume not aligned bars
        if len(coordinates) == 0:
            return 0

        # Scan through all the coordinates
        for x0, y0 in coordinates:
            
            y = tensor.shape[0] * x0
            x = tensor.shape[1] * y0
            img = Image.fromarray(tensor)

            # Extract patch
            patch = utils.extract_patch(img, x, y, config.PATCH_SIZE)
            # Normalize brightness
            # patch = utils.normalize_brightness_PIL(patch)
            tensor_patch = self.transform(patch)

            prediction = self.classifier.predict(tensor_patch.unsqueeze(0))

            # One not aligned bar is enough to classify the image as not aligned
            if prediction == 0:
                return 0
            
        return 1

    def evaluate(self, test_data, plot_confusion_matrix=True):
        predictions = []
        labels = []
        for img, label, _ in test_data:
            pred = self.predict(img)
            predictions.append(pred)
            labels.append(label)
        predictions = torch.Tensor(predictions)
        labels = torch.Tensor(labels)
        acc = self.accuracy(predictions, labels)
        fb = self.fb(predictions, labels)
        prec = self.precision(predictions, labels)
        rec = self.recall(predictions, labels)
        if plot_confusion_matrix:
            utils.plot_confusion_matrix(predictions, labels)
        return acc, fb, prec, rec
