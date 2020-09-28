""" FastEstimator Apphub parser """
import glob
import inspect
import json
import os
import pydoc
import re
import subprocess
import sys
import tempfile
from functools import partial
from shutil import copy

re_sidebar_title = '[^A-Za-z0-9:!,$%.() ]+'
re_route_title = '[^A-Za-z0-9 ]+'
re_url = '\[notebook\]\((.*)\)'


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def replacePath(match, path_prefix):
    tag = match.group(1)
    url = match.group(2)

    path = os.path.join(path_prefix, url)
    output = '![{}]({})'.format(tag, path)
    return output


def replaceImagePath(mdfile, d, branch):
    mdcontent = open(mdfile).readlines()
    re_image_path = r'!\[(.+)\]\((?!http)[\./]*(.+)\)'
    png_tag = '![png]('
    html_img_tag = '<img src="'
    path_prefix = os.path.join(f'assets/branches/{branch}/example', d)
    mdfile_updated = []

    for line in mdcontent:
        line = re.sub(re_image_path,
                      partial(replacePath, path_prefix=path_prefix), line)
        idx1, idx2 = map(line.find, [png_tag, html_img_tag])
        if idx2 != -1 and line.split(os.path.sep)[0] != 'assets':
            line = html_img_tag + os.path.join(path_prefix,
                                               line[idx2 + len(html_img_tag):])
            mdfile_updated.append(line)
        else:
            mdfile_updated.append(line)
    with open(mdfile, 'w') as f:
        f.write("".join(mdfile_updated))
    return mdfile


def extractTitle(md_path, fname):
    headers = ['#']
    f = os.path.join(md_path, fname + '.md')
    mdfile = open(f).readlines()
    for sentence in mdfile:
        sentence = sentence.strip()
        sentence_tokens = sentence.split(' ')
        if sentence_tokens[0] in headers:
            title = re.sub(re_sidebar_title, '', sentence)
            return title


def extractReadMe(output_path, apphub_path):
    readmefile = os.path.join(apphub_path, 'README.md')
    content = open(readmefile).readlines()
    toc = []
    startidx = 0
    endidx = len(content) - 1
    apphub_toc_path = os.path.join(output_path, 'overview.md')
    for i, line in enumerate(content):
        idx = line.find('Table of Contents:')
        if idx != -1:
            startidx = i
        elif line.split(' ')[0] == '##' and startidx != 0:
            endidx = i
            break

    overview = content[:startidx - 1]
    toc = content[startidx:endidx]

    # write table of content to the file
    with open(apphub_toc_path, 'w') as f:
        f.write("".join(overview))

    ex_list = []
    title_dict = {}
    child, files_list = [], []
    flag = False
    for line in toc[1:]:
        line_tokens = line.split(' ')
        if line_tokens[0] in ['###', '####']:
            if flag:
                title_dict['children'] = child
                ex_list.append(title_dict)
                child = []
                title_dict = {}
            title = re.sub(re_sidebar_title, '', line).strip()
            title_dict['title'] = title
        else:
            flag = True
            l = line.split(':')[0]
            name = re.sub(re_sidebar_title, '', l)
            url = re.findall(re_url, line)
            if name != '' and url:
                fname = os.path.basename(url[0]).split('.')[0]
                child_dict = {}
                child_dict[fname] = {'title': title, 'name': name.strip()}
                files_list.append(child_dict)
                child.append(child_dict)
    return files_list


def generateMarkdowns(apphub_path, output_path):
    json_struct = []
    exclude_prefixes = ['_', '.']
    for subdirs, dirs, files in os.walk(apphub_path, topdown=True):
        dirs[:] = [d for d in dirs if not d[0] in exclude_prefixes]
        for f in files:
            fname, ext = os.path.splitext(os.path.basename(f))
            example_type = splitall(os.path.relpath(subdirs, apphub_path))[0]
            save_dir = os.path.join(output_path, example_type)
            if ext == '.ipynb' and f[0] != '.':
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                subprocess.run([
                    'jupyter', 'nbconvert', '--to', 'markdown',
                    os.path.join(subdirs, f), '--output-dir', save_dir
                ])
            elif f.endswith(('png', 'jpeg', 'jpg')):
                filepath = os.path.join(subdirs, f)
                rel_filepath = os.path.relpath(filepath, apphub_path)
                image_subdir_elem = rel_filepath.split(os.path.sep)[2:-1]
                if len(image_subdir_elem) >= 1:
                    image_subdir = os.path.join(*image_subdir_elem)
                    output_image_path = os.path.join(save_dir, image_subdir)
                    if not os.path.exists(output_image_path):
                        os.makedirs(output_image_path)
                    copy(filepath, output_image_path)
                else:
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    copy(filepath, save_dir)


def getNameTitle(namedict, fname):
    for obj in namedict:
        if fname in obj.keys():
            title = obj[fname]['title']
            name = obj[fname]['name']
            return title, name


def create_json(output_path, apphub_path):
    json_dict = {}
    json_struct = []

    exclude_prefixes = ['_', '.']
    namedict = extractReadMe(output_path, apphub_path)
    for d in os.listdir(output_path):
        child_list = []
        parent_json_obj = {}
        if os.path.isdir(os.path.join(output_path, d)):
            files = [
                f for f in os.listdir(os.path.join(output_path, d))
                if os.path.isfile(os.path.join(*[output_path, d, f]))
            ]
            for f in files:
                file_json_obj = {}
                fname, ext = os.path.splitext(os.path.basename(f))
                if ext == '.md' and f[0] != '.':
                    title, name = getNameTitle(namedict, fname)
                    file_json_obj['name'] = os.path.join(d, fname + ext)
                    file_json_obj['displayName'] = name
                    child_list.append(file_json_obj)
            parent_json_obj['displayName'] = title
            parent_json_obj['name'] = d
            parent_json_obj['children'] = sorted(child_list, key=lambda x: x['displayName'])
            json_struct.append(parent_json_obj)
            
    json_struct = sorted(json_struct, key=lambda x: x['displayName'])
    json_struct.insert(0, {'displayName': 'Overview', 'name': 'overview.md'})
    return json_struct


if __name__ == '__main__':
    # take fastestimator dir path and output dir
    FE_DIR = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    branch = sys.argv[3]

    example_output_path = os.path.join(OUTPUT_PATH, 'example')
    apphub_path = os.path.join(FE_DIR, 'apphub')

    generateMarkdowns(apphub_path, example_output_path)
    for subdirs, dirs, files in os.walk(example_output_path, topdown=True):
        for f in files:
            if f.endswith('.md'):
                d = subdirs.split(os.path.sep)[-1]
                replaceImagePath(os.path.join(subdirs, f), d, branch)

    struct_json_path = os.path.join(example_output_path, 'structure.json')
    # write to json file
    with open(struct_json_path, 'w') as f:
        f.write(json.dumps(create_json(example_output_path, apphub_path)))
