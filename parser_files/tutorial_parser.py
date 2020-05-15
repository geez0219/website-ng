""" FastEstimator Docstring parser """
import json
import os
import re
import subprocess
import sys
from shutil import copy
from functools import partial

RE_SIDEBAR_TITLE = '[^A-Za-z0-9:!,$%. ]+'
RE_ROUTE_TITLE = '[^A-Za-z0-9 ]+'


def generateMarkdowns(output_dir, tutorial_type, fe_path):
    # form a input files path
    tutorial_sub_dir = os.path.join('tutorial', tutorial_type)
    tutorial_path = os.path.join(fe_path, tutorial_sub_dir)

    # invoke subprocess to run nbconvert command on notebook files
    output_sub_dir = os.path.join(*[output_dir, 'tutorial', tutorial_type])
    for filename in os.listdir(tutorial_path):
        if filename.endswith('.ipynb'):
            filepath = os.path.join(tutorial_path, filename)
            subprocess.run(['jupyter', 'nbconvert', '--to', 'markdown', filepath, '--output-dir', output_sub_dir])
        elif filename.endswith('.md'):
            filepath = os.path.join(tutorial_path, filename)
            copy(filepath, output_sub_dir)
    return output_sub_dir


def replaceRefLink(match, tutorial_type):
    ref_tutorial_type = match.group(1)
    tutorial_no = match.group(2)
    tutorial_name = match.group(3).split('/')
    if ref_tutorial_type != '':
        tutorial = '[{} tutorial {}]'.format(ref_tutorial_type, tutorial_no)
    else:
        tutorial = '[tutorial {}]'.format(tutorial_no)
    if len(tutorial_name) > 1:
        return '{}(./tutorials/{}/{})'.format(tutorial, tutorial_name[-2], tutorial_name[-1])
    else:
        return '{}(./tutorials/{}/{})'.format(tutorial, tutorial_type, tutorial_name[-1])


def replaceApphubLink(match):
    apphub_link_segments = match.group(3).strip().split('/')
    dir_name = apphub_link_segments[1]
    name = apphub_link_segments[-1]
    return '[{}](./examples/{}/{})'.format(match.group(1), dir_name, name)


def replaceAnchorLink(match, tutorial_type, fname):
    anchor_text = match.group(1)
    anchor_link = match.group(2)
    fname = fname.split('/')[-1].split('.')[0]
    return '[{}](./tutorials/{}/{}#{})'.format(anchor_text, tutorial_type, fname, anchor_link)


def replaceRepoLink(match):
    name = match.group(1)
    url = match.group(2)
    fe_url = 'https://github.com/fastestimator/fastestimator/tree/r1.0/'
    return '[{}]({})'.format(name, os.path.join(fe_url, url))


def updateLinks(line, tutorial_type, fname):
    re_ref_link = r'\[(\w*)\s*[tT]utorial\s*(\d+)\]\(\.+\/([^\)]*)\.ipynb\)'
    re_apphub_link = r'\[([\w\d\-\/\s]+)\]\((.+apphub(.+))\.ipynb\)'
    re_anchortag_link = r'\[([^\]]+)\]\(#([^)]+)\)'
    re_repo_link = r'\[([\w|\d|\s]*)\]\([\./]*([^\)\.#]*)(?:\.py|)\)'

    output_line = re.sub(re_ref_link, partial(replaceRefLink, tutorial_type=tutorial_type), line)
    output_line = re.sub(re_apphub_link, replaceApphubLink, output_line)
    output_line = re.sub(re_anchortag_link, partial(replaceAnchorLink, tutorial_type=tutorial_type,
                                                    fname=fname), output_line)
    output_line = re.sub(re_repo_link, replaceRepoLink, output_line)

    return output_line


def replaceImagePath(mdfile, tutorial_type):
    """This function takes markdown file path and append the prefix path to the image path in the file. It allows
    angular to find images in the server.

    Args:
        mdfile: markdown file
        tutorial_type = Type of tutorial e.g. 'beginner' or 'advanced'
    """
    mdcontent = open(mdfile).readlines()
    png_tag = '![png]('
    html_img_tag = '<img src="'
    path_prefix = 'assets/branches/r1.0/tutorial'
    png_path_prefix = 'assets/branches/r1.0/tutorial/' + tutorial_type
    mdfile_updated = []
    for line in mdcontent:
        line = updateLinks(line, tutorial_type, mdfile)
        idx1, idx2 = map(line.find, [png_tag, html_img_tag])
        if idx1 != -1:
            line = png_tag + os.path.join(png_path_prefix, line[idx1 + len(png_tag):])
            mdfile_updated.append(line)
        elif idx2 != -1:
            line = html_img_tag + os.path.join(path_prefix, line[idx2 + len(html_img_tag):])
            mdfile_updated.append(line)
        else:
            mdfile_updated.append(line)
    with open(mdfile, 'w') as f:
        f.write("".join(mdfile_updated))


def createJson(output_dir):
    dir_arr = []
    headers = ['#']
    subheaders = ['##', '###']
    tutorial_output_path = os.path.join(output_dir, 'tutorial')
    for f in os.scandir(tutorial_output_path):
        if f.is_dir():
            dir_obj = {}
            dir_obj['name'] = f.name
            dir_obj['displayName'] = f.name.capitalize() + ' Tutorials'
            children = []
            for filename in sorted(os.listdir(f)):
                if filename.endswith('.md'):
                    filepath = os.path.join(f, filename)
                    # replace ref links in the markdown files
                    replaceImagePath(filepath, f.name)
                    # open updated markdown file and extract table of content
                    mdfile = open(os.path.join(f, filename)).readlines()
                    flag = True
                    f_obj = {}
                    sidebar_titles = []
                    for sentence in mdfile:
                        sentence = sentence.strip()
                        sentence_tokens = sentence.split(' ')
                        sidebar_val_dict = {}
                        if flag and sentence_tokens[0] in headers:
                            f_obj['name'] = os.path.join(f.name, filename.split('.')[0])
                            f_obj['displayName'] = re.sub(RE_SIDEBAR_TITLE, '', sentence)
                            flag = False
                        elif sentence_tokens[0] in subheaders:
                            title = re.sub(RE_SIDEBAR_TITLE, '', sentence)
                            route_title = re.sub(RE_ROUTE_TITLE, '', sentence)
                            sidebar_val_dict['id'] = route_title.lower().strip().replace(' ', '-')
                            sidebar_val_dict['displayName'] = title.strip()
                            sidebar_titles.append(sidebar_val_dict)
                    f_obj['toc'] = sidebar_titles
                    children.append(f_obj)
            dir_obj['children'] = children
        dir_arr.append(dir_obj)

    output_struct = os.path.join(tutorial_output_path, 'structure.json')
    # write to json file
    with open(output_struct, 'w') as f:
        f.write(json.dumps(dir_arr))


if __name__ == '__main__':
    # take fastestimator dir path and output dir
    FE_DIR = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]

    # generate markdown to temp dir
    beginner_output_path = generateMarkdowns(OUTPUT_PATH, 'beginner', FE_DIR)
    advanced_output_path = generateMarkdowns(OUTPUT_PATH, 'advanced', FE_DIR)
    createJson(OUTPUT_PATH)
