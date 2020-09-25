## Prerequisites:
* Python >= 3.6

## Installation:
### 1. Install Dependencies:
* Windows (CPU):
    ``` bash
    $ pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```

* Linux (CPU/GPU):
    ``` bash
    $ apt-get install libglib2.0-0 libsm6 libxrender1 libxext6
    ```

* Mac (CPU):
    ``` bash
    $ echo No dependency needed ":)"
    ```

### 2. Install FastEstimator:
* Stable:
    ``` bash
    $ pip install fastestimator
    ```
* Most Recent:
    ``` bash
    $ pip install fastestimator-nightly
    ```


## Docker Hub
Docker containers create isolated virtual environments that share resources with a host machine. Docker provides an easy way to set up a FastEstimator environment. You can simply pull our image from [Docker Hub](https://hub.docker.com/r/fastestimator/fastestimator/tags) and get started:

* GPU:
    ``` bash
    docker pull fastestimator/fastestimator:latest-gpu
    ```
* CPU:
    ``` bash
    docker pull fastestimator/fastestimator:latest-cpu
    ```