import torch
import torch.nn.functional as F
from torchmetrics.classification import BinaryAccuracy, FBetaScore, Precision, Recall, ROC
import lightning as L
from utils import _get_flattened_size
from torch import nn
from torchsummary import summary
import config
from torchvision.models import resnet50, ResNet50_Weights
import torchmetrics


class PatchClassifier(L.LightningModule):

    def __init__(self, feature_extractor=resnet50(weights=ResNet50_Weights.DEFAULT), learning_rate=1e-2):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.feature_extractor.eval()
        self.learning_rate = learning_rate
        self.confusion_matrix = torchmetrics.ConfusionMatrix(task='binary')

        # Freeze the feature extractor parameters
        for param in self.feature_extractor.parameters():
            param.requires_grad = False

        in_size = _get_flattened_size(self.feature_extractor, torch.randn(1, 3, config.PATCH_SIZE, config.PATCH_SIZE))

        self.classifier = nn.Sequential(
            # nn.Flatten(),
            nn.Linear(in_size, 500), 
            nn.ReLU(),
            nn.Linear(500, 200),
            nn.ReLU(),
            nn.Linear(200, 48),
            nn.ReLU(),
            nn.Linear(48, 24),
            nn.ReLU(),
            nn.Linear(24, 2)
        )

        self.fb = FBetaScore(task="binary", beta=0.5, threshold=0.7)
        self.recall = Recall(task="binary", num_classes=2)
        self.precision = Precision(task="binary", num_classes=2)
        self.roc = ROC(task="binary", num_classes=2)
        self.accuracy = BinaryAccuracy()

    def forward(self, x):
        with torch.no_grad():
            x = self.feature_extractor(x)
        return self.classifier(x)
    
    def training_step(self, batch, batch_idx):

        loss, acc, fb, prec, rec, _ = self.common_step(batch, batch_idx)

        metrics = {'train_acc': acc, 'train_loss': loss, 'train_fbeta': fb
                   , 'train_precision': prec, 'train_recall': rec}

        self.log('train_loss', loss)
        self.log('train_acc', acc)
        self.log('train_fbeta', fb)
        self.log('train_precision', prec)
        self.log('train_recall', rec)
        self.log_dict(metrics)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, acc, fb, prec, rec, _  = self.common_step(batch, batch_idx)
        metrics = {'val_acc': acc, 'val_loss': loss, 'val_fbeta': fb, 
                   'val_precision': prec, 'val_recall': rec}
        self.log('val_loss', loss)
        self.log('val_acc', acc)
        self.log('val_fbeta', fb)
        self.log('val_precision', prec)
        self.log('val_recall', rec)
        self.log_dict(metrics)
        return metrics

    def common_step(self, batch, batch_idx):
        x, y = batch
        y_prob = self.forward(x)
        y = y.long()
        loss, acc, fb, prec, rec, y_prob = self.compute_metrics(y, y_prob)

        return loss, acc, fb, prec, rec, y_prob

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def model_summary(self):
        summary(self, input_size=(3, 576, 864))

    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_prob = self.forward(x)
        y = y.long()
        loss, acc, fb, prec, rec, y_prob = self.compute_metrics(y, y_prob)

        self.log('test_loss', loss)
        self.log('test_acc', acc)
        self.log('test_fbeta', fb)
        self.log('test_precision', prec)
        self.log('test_recall', rec)

        return {'test_loss': loss, 'test_acc': acc, 'test_fbeta': fb,
                'test_precision': prec, 'test_recall': rec}

    def on_validation_end(self):
        conf_matrix = self.confusion_matrix.compute()
        print("Confusion Matrix:\n", conf_matrix)

    def compute_metrics(self, y_true, y_prob):
        loss = F.cross_entropy(y_prob, y_true)
        y_prob = y_prob.argmax(dim=1)
        acc = self.accuracy(y_prob, y_true)
        fb = self.fb(y_prob, y_true)
        prec = self.precision(y_prob, y_true)
        rec = self.recall(y_prob, y_true)

        return loss, acc, fb, prec, rec, y_prob