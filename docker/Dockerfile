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
    git \
    wget \
    unzip

RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt-get update
RUN apt-get install python3.6 -y --no-install-recommends
RUN curl https://bootstrap.pypa.io/get-pip.py get-pip.py | python3.6

RUN python3.6 -m pip install --no-cache-dir \
    bs4 \
    selenium \
    lxml

# configure node
ENV NVM_DIR /root/.nvm
ENV NODE_VERSION 14.3.0

RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.36.0/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

ENV NODE_PATH $NVM_DIR/versions/node/v$NODE_VERSION/lib/node_modules
ENV PATH      $NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# RUN apt-get install -y chromium-browser

# configure chrome for scraper
# Adding trusting keys to apt for repositories
RUN wget -q -O - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add -

# Adding Google Chrome to the repositories
RUN sh -c 'echo "deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main" >> /etc/apt/sources.list.d/google-chrome.list'

# Updating apt to see and install Google Chrome
RUN apt-get -y update

# Magic happens
RUN apt-get install -y google-chrome-stable

# Download the Chrome Driver
RUN wget -O /tmp/chromedriver.zip http://chromedriver.storage.googleapis.com/` \
curl -sS chromedriver.storage.googleapis.com/LATEST_RELEASE \
`/chromedriver_linux64.zip

ENV CHROMEDRIVER_DIR /usr/local/bin/

# Unzip the Chrome Driver into /usr/local/bin directory
RUN unzip /tmp/chromedriver.zip chromedriver -d /usr/local/bin/