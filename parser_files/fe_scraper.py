'''FastEstimator elasticsearch content parsing script. This script download and parse the search content from the fastestimator locally hosted
website.'''
import json
import os
import re
import time
import urllib.request as urllib2
from urllib.parse import urljoin
import pdb
from bs4 import BeautifulSoup
from selenium import webdriver

#fastestimator url to append
FE_URL = 'https://www.fastestimator.org'
EXAMPLES_DIR = 'examples'
API_DIR = 'api'
TUTORIALS_DIR = 'tutorials'
INSTALL_DIR = 'install'
MAIN_DIR = 'main'

#change this to stage env for crawling in pipeline
LOCAL_URL = 'http://localhost:4200'

#initialize the selenium driver for the chrome
options = webdriver.ChromeOptions()
options.add_argument('--ignore-certificate-errors')
options.add_argument('--headless')
driver = webdriver.Chrome(
    executable_path="/home/geez219/Downloads/chromedriver",
    chrome_options=options)


def clean_body(text):
    s = re.sub(r"\s\s+", " ", text)
    s = s.replace('\n', ' ')
    return s


def save_json_file(fname, parent_dir, item):
    #create directory if it doesnt exsit and save file
    if not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(os.path.join(parent_dir, fname), 'w') as f:
        f.write(json.dumps(item))


def extract_examples(url):
    links = []
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')
    nodes = soup.find_all('mat-tree-node')

    for node in nodes:
        a = node.find('a')
        links.append(a.attrs['href'])

    for link in links:
        item = {}
        driver.get(LOCAL_URL + link)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        markdown = soup.find('markdown')
        h1 = soup.find('h1')

        item['link'] = FE_URL + link
        item['body'] = clean_body(markdown.text)
        item['title'] = h1.text

        #save json file
        fname = item['link'].split('/')[-1] + '.json'
        save_json_file(fname, EXAMPLES_DIR, item)


def extract_tutorial(url):
    links = []
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')
    nodes = soup.find_all('mat-tree-node')
    for node in nodes:
        a = node.find('a')
        links.append(a.attrs['href'])
    print(links)

    for link in links:
        print(link)
        pdb.set_trace()
        item = {}
        driver.get(LOCAL_URL + link)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        markdown = soup.find('markdown')
        h1 = soup.find('h1')

        item['link'] = FE_URL + link
        item['body'] = clean_body(markdown.text)
        title = h1.text
        item['title'] = title

        #save json file
        fname = item['link'].split('/')[-1] + '.json'
        save_json_file(fname, TUTORIALS_DIR, item)


def extract_api(url):
    links = []
    driver.get(url)
    page_source = driver.page_source
    soup = BeautifulSoup(page_source, 'lxml')
    nodes = soup.find_all('mat-tree-node')
    for node in nodes:
        a = node.find('a')
        links.append(a.attrs['href'])
    print(links)

    for link in links:
        item = {}
        driver.get(LOCAL_URL + link)
        soup = BeautifulSoup(driver.page_source, 'lxml')
        markdown = soup.find('markdown')

        item['link'] = FE_URL + link
        item['body'] = clean_body(markdown.text)
        title = item['link'].split('/')[-1]
        dirname = item['link'].split('/')[-2]
        item['title'] = title

        #save json file
        fname = title + '.json'
        save_json_file(dirname + '_' + fname, API_DIR, item)


def extract_install(url):
    driver.get(url)
    item = {}
    soup = BeautifulSoup(driver.page_source, 'lxml')
    div = soup.find('div', {'class': 'content'})
    item['link'] = FE_URL + '/install'
    item['body'] = clean_body(div.text)
    item['title'] = 'Install'

    fname = item['link'].split('/')[-1] + '.json'
    save_json_file(fname, INSTALL_DIR, item)


def extract_main(url):
    driver.get(url)
    item = {}
    soup = BeautifulSoup(driver.page_source, 'lxml')
    div = soup.find('div', {'class': 'container'})

    item['link'] = FE_URL
    item['body'] = clean_body(div.text)
    item['title'] = 'Getting Started'

    fname = 'gettingstarted.json'
    save_json_file(fname, MAIN_DIR, item)


'''
FUTURE SETTING: Initial point to follow outbound anchor tags in multiple depths and crawl
HTML pages on the way. Addtionally, crawler needs to identify specific page and store it in the
directory.
'''


def extract_main_list(url):
    driver.get(url)
    soup = BeautifulSoup(driver.page_source, 'lxml')
    ul = soup.find_all('li', {'class': 'nav-item'})
    for li in ul:
        a = li.find('a')


if __name__ == '__main__':
    #relative urls
    example_rel_url = 'examples/overview'
    tutorial_rel_url = 'tutorials/beginner/t01_getting_started'
    api_rel_url = 'api/fe/Estimator'
    install_rel_url = 'install'

    example_url = urljoin(LOCAL_URL, example_rel_url)
    tutorial_url = urljoin(LOCAL_URL, tutorial_rel_url)
    api_url = urljoin(LOCAL_URL, api_rel_url)
    install_url = urljoin(LOCAL_URL, install_rel_url)

    # extract_examples(example_url)
    extract_tutorial(tutorial_url)
    # extract_api(api_url)
    # extract_install(install_url)
    # extract_main(LOCAL_URL)
    #extract_main_list(main_url)