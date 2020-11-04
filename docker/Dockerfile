FROM ubuntu:16.04

# Replace shell with bash so we can source files
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# configure python3.6
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
    software-properties-common \
    python-software-properties \
    curl \
    apt-utils \
    git

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 -y --no-install-recommends
RUN curl https://bootstrap.pypa.io/get-pip.py get-pip.py | python3.6

RUN python3.6 -m pip install --no-cache-dir \
    bs4 \
    selenium

# configure node
ENV NVM_DIR /root/.nvm

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.36.0/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install 14.3.0