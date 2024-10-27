# Automated Detection of Steel Bars Using Machine Vision and Deep Learning

## Authors
- [Aliprandi Francesco](https://github.com/francealip)
- [Degiorgi Nicola](https://github.com/ssbarbaro)
- [De Castelli Fabrizio](https://github.com/FabriDeCastelli)
- [Loika Arseni](https://github.com/loikaar)
- [Sbarbaro Steven](https://github.com/xflappy)

## Video
https://polybox.ethz.ch/index.php/s/RKFzqX2ubVlIgmQ

## Abstract
This project performs binary classification of steel beam bars by combining machine vision techniques with deep learning model. First task involves data cleaning using machine vision approaches. Next, a convolution-deconvolution architecture is employed for key corner detection, followed by an image classifier. Both corner detection and classification utilize pre-trained models from TorchVision.

## Running the code
For running tests you just need to execute```inference.py``` that takes an argument ```-s path/to/dataset``` to specify the data directory. 

You can install all required dependencies with:

```bash
python /teamspace/studios/this_studio/inference.py -s path_to_dataset
```

## Requirements
To run this project, you will need Python and all the dependecies in ```requirements.txt```

You can install all required dependencies with:

```bash
pip install -r requirements.txt
```
## Project Directory Structure

```
THIS_STUDIO/
├── .lightning_studio/
│   ├── on_start.sh
│   └── on_stop.sh
├── dataset/
│   ├── export1.csv
│   ├── export2.csv
│   ├── export3.csv
│   └── test_set.csv
├── lightning_logs/
│   └── performance_final/
│       └── events.out.tfevents.1730021584.ip-10-19-90-222
├── models/
│   ├── Final_classifier.pth
│   └── gaussian_points_finder.pth
├── plots/
│   ├── confusion_matrix.png
│   ├── Filtered.jpg
│   ├── Filtered2.jpg
│   ├── rgb.jpg
│   └── test.jpg
├── .gitignore
├── config.py
├── dataloader.py
├── dataset_access.txt
├── gaussian_point_finder.py
├── guidelines.pdf
├── inference.py
├── LICENSE.md
├── patch_classifier.py
├── point_finder_training.ipynb
├── predictor.py
├── Project_Proposal_Duferco.pdf
├── README.md
└── requirements.txt
```

