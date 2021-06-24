# FasEstimator website

## Pipeline to deploy website
The FastEstimator website has a pipeline to handle every stage of deployment including:
1. parsing FE online repo
2. building web application
3. scraping web page (collecting search index)
4. search server update
5. website deployment

The pipeline is built on Jenkins (http://jenkins.fastestimator.org:8080/job/fastestimator_web/) and its Jenkinsfile is
in the `CICD` folder.

---

## Parsing FE
```
python parser_files/parse_all.py <fe_path> <output_path> <branch>
```
This will parse the FE repo at \<fe_path\> and generates the assets file dir at \<output_path\>. \<branch\> is the
git branch of that FE repo.

---

## Building Website

### Configure
* step 0: build docker image
    ```
    docker build -t <image> - < docker/Dockerfile
    ```

* step 1: build docker environment (`--network host` is required for localhost mapping)
    ```
    docker run -it --network host <image>
    ```
* step 2: install npm dependency
    ```
    npm install
    ```

### Test

* serve the website on localhost
    ```
    npm run ng serve
    ```

* serve the website on localhost (SSR)
    ```
    npm run dev:ssr
    ```

### Build

* build the website (SSR)
    ```
    npm run build:ssr
    ```

---

## Scraping Website

Scraping website means crawl the website and generate the mapping of keyword to content for search server usage

* step 1: configure Chrome and download the Chrome Driver (docker/Dockerfile cover all)
* step 2: serve the website (ssr)
* step 3: run website_scraping
    ```
    python parsed_files/fe_scraper.py <branch> <index_dir> <chrome_driver_path>
    ```
  This will generate index folder at \<index_dir\>

---

## Search server update
Our website has own search server, therefore every time before we depoly the new website, we need to update the
search server. Search server is set up in ec2 instance `FE-Search(don't stop)`. The server repo is https://github.com/fastestimator-util/nodejs-elasticsearch

* step 1: copy the \<index_dir\> generated at scraping stage to `FE-Search(don't stop)`

* step 2: run load_index.py at `FE-Search(don't stop)` instance for each branch
    ```
    python3 load_index.py <index_dir>/<branch> <branch>
    ```

---

## Deployment (manually)

* step 1: Build the website (ssr)
* step 2: Compress the dist/ and package.json into a zip file

    example code:
    ```
    zip -r dist.zip dist package.json
    ```
* step 3: Upload the zip file to AWS elastic beanstalk