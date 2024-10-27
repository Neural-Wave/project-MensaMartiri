# MENSA MARTIRI DUFERCO PROJECT
# 
# Inference script for the model

import argparse
from predictor import Predictor
import utils
import time
import torch

if __name__ == "__main__":    
    
    parser = argparse.ArgumentParser(description="Test argument parsing")
    parser.add_argument("-s", "--dataset", type=str, required=True, help="Path to dataset directory")
    args = parser.parse_args()

    dataset_path = args.dataset 
    predictor = Predictor(model_path="models/final_classifier.pth")
    
    test_data = utils.get_test_data(test_set_directory=dataset_path)

    # FOR SINGLE DATA PREDICTION, UNCOMMMENT THE FOLLOWING LINES
    """
    for img, label, _ in test_data:
        prediction = predictor.predict(img)
        print(f"Prediction: {prediction}, Label: {label}")
    """

    print("\nRunning tests on directory\n", dataset_path)

    # print in which device i am running using cuda
    print("Device: ", torch.cuda.get_device_name(0))

    # EVALUATION: set the parmeter 'plot_confusion_matrix=False' to disable plotting 
    timer_start = time.time()
    acc, fb, prec, rec = predictor.evaluate(test_data)
    timer_end = time.time()
    print("Accuracy: ", acc.item(), "F-beta: ", fb.item(), "Precision: ", prec.item(), "Recall: ", rec.item())
    print(f"Time elapsed:  {round(timer_end - timer_start, 2)}s")

