# Ｍｏｄｅｌ Cᴏᴍᴘʀᴇꜱꜱɪᴏɴ

Compress given pretrained model on ImageNet, by applying Pruning, Knowledge Distillation and Quantization. The compression will reduce the resource requirements (memory and GFLOPS) enabling the model to be used in low-resource environments like edge device, phone, autonomous vechiles etc. In compressing a model, we effectively reduce the carbon footprint, especially at scale. See benchmark section for specific results. 

Comes with object-detection demo. 

![](https://github.com/rpa020/compression/blob/main/images/od.gif)

# Content:

* Compression Application:

`-` prune.py -> Pruning program
python prune.py *arguments "/path/to/imagenet"

`-` distillation.py -> Knowledge Distillation program
python distillation.py *arguments "/path/to/imagenet"

`-` quantization.py -> Quantization program
python quantization.py *arguments "/path/to/imagenet"

`-` compresser.py -> Compresser application running all 3 mentioned programs
python compresser.py *arguments "/path/to/imagenet"

`-` UI.py -> Simple user interface of compression application
python UI.py

`-` utils.py -> model-preparer, train_one_epoch, validate, etc.

`-` acc.txt -> records validation accuracy

`-` requirements.txt -> Lists of requirements. Application is run with python 3.8.13

* Object-Detection:

`-` demo.py -> Run the demo application `python demo.py`

`-` labels.py -> List over classes, used to classify in object detection

`-` model.pth -> The model used in object detection


* Models: Already compressed models

# How to run compression application:

`python compresser.py --sa resnet18 --ta resnet152 "/path/to/imagenet"` or `python UI.py`.
After compressing, the available model can be loaded with torch.jit.load("model.ts")

# Benchmark
All models are compressed with default values. 

![](https://github.com/rpa020/compression/blob/main/images/accuracy.png)
![](https://github.com/rpa020/compression/blob/main/images/reduction.png)

When running with a new model, make sure to testurn with 0 epochs for each compression method to testrun and observe if it's compressible with this application. 

