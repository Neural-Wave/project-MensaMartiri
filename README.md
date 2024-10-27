# Automated Detection of Steel Bars Using Machine Vision and Deep Learning

## Authors
- [Aliprandi Francesco](https://github.com/francealip)
- [Degiorgi Nicola](https://github.com/ssbarbaro)
- [De Castelli Fabrizio](https://github.com/FabriDeCastelli)
- [Loika Arseni](https://github.com/loikaar)
- [Sbarbaro Steven](https://github.com/xflappy)


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

