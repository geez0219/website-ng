## Prerequisites:
* Python >= 3.6

## Installation:
### 1. Install Dependencies:
* Windows (CPU):
    ``` bash
    $ pip install torch==1.4.0+cpu torchvision==0.5.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
    ```
    * Install Visual C++ 2015 build tools [here](https://go.microsoft.com/fwlink/?LinkId=691126) and install default option.

    * Install latest Visual C++ redistributable [here](https://support.microsoft.com/en-us/help/2977003/the-latest-supported-visual-c-downloads) and choose x86 for 32 bit OS, x64 for 64 bit OS.

* Linux (CPU/GPU):
    ``` bash
    $ apt-get install libglib2.0-0 libsm6 libxrender1 libxext6
    ```

* Mac (CPU):
    ``` bash
    $ echo No dependency needed ":)"
    ```

### 2. Install FastEstimator:
* Stable (Linux/Mac):
    ``` bash
    $ pip install fastestimator
    ```

* Stable (Windows):

    First download zip file from [available releases](https://github.com/fastestimator/fastestimator/releases)
    ``` bash
    $ pip install fastestimator-x.x.x.zip
    ```

* Most Recent (Linux/Mac):
    ``` bash
    $ pip install fastestimator-nightly
    ```

* Most Recent (Windows):

    First download zip file [here](https://github.com/fastestimator/fastestimator/archive/master.zip)
    ``` bash
    $ pip install fastestimator-master.zip
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