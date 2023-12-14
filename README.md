
#【M】【o】【d】【e】【l】
# ℭ𝔬𝔪𝔭𝔯𝔢𝔰𝔰𝔦𝔬𝔫

Code for compressing given pretrained model on ImageNet.
Applies Pruning, Knowledge Distillation and Quantization.
Comes with object-detection demo. 


# Contents:

* Compression Application:

- prune.py -> Pruning program
python prune.py *arguments /path/to/imagenet

- distillation.py -> Knowledge Distillation program
python distillation.py *arguments /path/to/imagenet

- quantization.py -> Quantization program
python quantization.py *arguments /path/to/imagenet

- compresser.py -> Compresser application running all 3 mentioned programs
python compresser.py *arguments /path/to/imagenet

- UI.py -> Simple user interface of compression application
python UI.py

- utils.py -> model-preparer, train_one_epoch, validate, etc.

- acc.txt -> records validation accuracy

- requirements.txt -> Lists of requirements. Application is run with python 3.8.13

* Object-Detection:


* Models: Already compressed models


# How to run compression application:

`python compresser.py --sa resnet18 --ta resnet152 "/path/to/imagenet"` or `python UI.py`.
After compressing, the available model can be loaded with torch.jit.load("model.ts)

# Benchmark

When running with a new model, make sure to testurn with 0 epochs for each compression method to testrun and observe if it's compressible with this application. 
