## Prerequisites:
* Python >= 3.6
* TensorFlow2

    * GPU:  `pip install tensorflow-gpu==2.0.0`
    * CPU:  `pip install tensorflow==2.0.0`

* Pytorch backend is coming soon!

## Installation
Please choose one:
* I have no idea what FastEstimator is about:
```
pip install fastestimator==1.0b2
```
* I want to keep up to date with the latest:
```
pip install fastestimator-nightly
```
* I'm here to play hardcore mode:

```
git clone https://github.com/fastestimator/fastestimator.git
pip install -e fastestimator
```


## Docker Hub
Docker container creates isolated virtual environment that shares resources with host machine. Docker provides an easy way to set up FastEstimator environment, users can pull image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags).

* GPU: `docker pull fastestimator/fastestimator:1.0b2-gpu`
* CPU: `docker pull fastestimator/fastestimator:1.0b2-cpu`

## Start your first FastEstimator training

```
$ python ./apphub/image_classification/lenet_mnist/lenet_mnist.py
```