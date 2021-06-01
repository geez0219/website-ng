""" FastEstimator tutorial parser. This script parses the FE tutorial and generates the assets files for webpage

    The example parsing case where...
    * FE_DIR = "fastestimator"
    * OUTPUT_PATH = "branches/r1.2/"

    (repo)                                  (asset)
    fastestimator/tutorial/                 branches/r1.2/
    ├── advanced/                           ├── tutorial/
    │   ├── t01_dataset.ipynb               │    ├── advanced/
    │   ├── t02_pipeline.ipynb              │    │   ├── t01_dataset.md
    │   └── ...                             │    │   ├── t02_pipeline.md
    ├── beginner/                           │    │   └── ...
    │   ├── t01_getting_started.ipynb   =>  │    ├── beginner/
    │   ├── t02_dataset.ipynb               │    │   ├── t01_getting_started.md
    │   └── ...                             │    |   ├── t01_getting_started_files/
    └── resources                           |    |   |   └─ t01_getting_started_19_1.png
        ├── t01_api.png                     │    |   ├── t02_dataset.md
        └── ...                             |    |   └── ...
                                            |    └── structure.json
                                            └── resources/
                                                ├── t01_api.png
                                                └── ...
                                            (Note: the "resources" dir is not generated from this parser, but the
                                             assets expects it. It will be copied by other parsing script)

    * It expect the source of tutorial dir has only two sub-dirs: "advanced" and "beginner" and all tutorial notebook
      files are exactly in those two dir levels (not under further nested dir).

"""
import json
import os
import pdb
import re
import shutil
import subprocess
import sys
from functools import partial
from shutil import copy

RE_SIDEBAR_TITLE = '[^A-Za-z0-9:!,$%. ()]+'
RE_ROUTE_TITLE = '[^A-Za-z0-9 ]+'
BRANCH = None


def generateMarkdowns(source_tutorial, output_tutorial):
    """ Parse repo tutorial and generate assets tutorial for webpage.
    Args:
        source_tutorial: tutorial source (ex: fastestimator/tutorial)
        output_tutorial: parsed tutorial destination (ex: r1.1/tutorial)
    """

    if os.path.exists(output_tutorial):
        shutil.rmtree(output_tutorial)

    for path, dirs, files in os.walk(source_tutorial):
        # not walk into the dir start with "_" and ".", and avoid going to the "resources" dir (this is reserved for
        # tutorial resources)
        dirs[:] = [
            d for d in dirs if d[0] not in ["_", "."] and d != "resources"
        ]

        output_dir = os.path.join(output_tutorial,
                                  os.path.relpath(path, source_tutorial))
        os.makedirs(output_dir, exist_ok=True)
        for f in files:
            if f.endswith('.ipynb'):
                filepath = os.path.join(path, f)
                # invoke subprocess to run nbconvert command on notebook files
                subprocess.run([
                    'jupyter', 'nbconvert', '--to', 'markdown', filepath,
                    '--output-dir', output_dir
                ])
            elif f.endswith('.md'):
                filepath = os.path.join(path, f)
                copy(filepath, output_dir)


def replaceRefLink(match, tutorial_type):
    """
    ex: [Tutorial 2](./t02_dataset.ipynb)
     -> [Tutorial 2](./tutorials/master/beginner/t02_dataset)
    """
    ref_tutorial_type = match.group(1)
    tutorial_no = match.group(2)
    tutorial_name = match.group(3).split('/')

    if ref_tutorial_type != '':
        tutorial = '[{} Tutorial {}]'.format(ref_tutorial_type, tutorial_no)
    else:
        tutorial = '[Tutorial {}]'.format(tutorial_no)
    if len(tutorial_name) > 1:
        return f'{tutorial}(./tutorials/{BRANCH}/{tutorial_name[-2]}/{tutorial_name[-1]})'
    else:
        return f'{tutorial}(./tutorials/{BRANCH}/{tutorial_type}/{tutorial_name[-1]})'


def replaceApphubLink(match):
    """
    ex: [MNIST](../../apphub/image_classification/mnist/mnist.ipynb)
     -> [MNIST](./examples/master/image_classification/mnist)
    """
    apphub_link_segments = match.group(3).strip().split('/')
    dir_name = apphub_link_segments[1]
    name = apphub_link_segments[-1]
    return f'[{match.group(1)}](./examples/{BRANCH}/{dir_name}/{name})'


def replaceAnchorLink(match, tutorial_type, fname):
    """
    ex: [Traces](#ta05trace)
     -> [Traces](./tutorials/master/advanced/t05_scheduler#ta05trace)
    """
    anchor_text = match.group(1)
    anchor_link = match.group(2)
    fname = fname.split('/')[-1].split('.')[0]
    return f'[{anchor_text}](./tutorials/{BRANCH}/{tutorial_type}/{fname}#{anchor_link})'


def replaceRepoLink(match):
    """
    ex: [Architectures](../../fastestimator/architecture)
     -> [Architectures](https://github.com/fastestimator/fastestimator/tree/{BRANCH}/fastestimator/architecture)
    """
    name = match.group(1)
    url = match.group(2)
    fe_url = f'https://github.com/fastestimator/fastestimator/tree/{BRANCH}/'
    return '[{}]({})'.format(name, os.path.join(fe_url, url))


def replaceImgLink(match):
    """
    deal with src=""
    ex: <img src="../resources/t01_api.png" alt="drawing" width="700"/>
     -> <img src="assets/branches/master/tutorial/../resources/t01_api.png" alt="drawing" width="700"/>
    """
    path_prefix = f'assets/branches/{BRANCH}/tutorial'
    new_src = os.path.join(path_prefix, match.group(1))

    return f'src=\"{new_src}\"'


def replaceImgLink2(match, tutorial_type):
    """
    deal with img link from papermill nbconvert generated markdown
    ex: ![png](t01_getting_started_files/t01_getting_started_19_1.png)
     -> ![png](assets/branches/master/tutorial/beginner/t01_getting_started_files/t01_getting_started_19_1.png)
    """
    return f'![png](assets/branches/{BRANCH}/tutorial/{tutorial_type}/{match.group(1)})'


def replaceLink(line, tutorial_type, fname):
    re_ref_link = r'\[(\w*)\s*[tT]utorial\s*(\d+)\]\(\.+\/([^\)]*)\.ipynb\)'
    re_apphub_link = r'\[([\w\d\-\/\s]+)\]\((.+apphub(.+))\.ipynb\)'
    re_anchortag_link = r'\[([^\]]+)\]\(#([^)]+)\)'
    re_repo_link = r'\[([\w|\d|\s]*)\]\([\./]*([^\)\.#]*)(?:\.py|)\)'
    re_src_img_link = r'src=\"([^ ]+)\"'
    re_png_img_link = r'!\[png\]\((.+)\)'

    output_line = re.sub(re_repo_link, replaceRepoLink, line)
    output_line = re.sub(
        re_ref_link, lambda x: replaceRefLink(x, tutorial_type=tutorial_type),
        output_line)
    output_line = re.sub(re_apphub_link, replaceApphubLink, output_line)
    output_line = re.sub(
        re_anchortag_link, lambda x: replaceAnchorLink(
            x, tutorial_type=tutorial_type, fname=fname), output_line)

    output_line = re.sub(re_src_img_link, replaceImgLink, output_line)
    output_line = re.sub(
        re_png_img_link,
        lambda x: replaceImgLink2(x, tutorial_type=tutorial_type), output_line)

    return output_line


def updateLink(mdfile, tutorial_type):
    """Replace all links to fit web application.

    Args:
        mdfile: markdown file path
        tutorial_type = Type of tutorial e.g. 'beginner' or 'advanced'
    """
    mdcontent = open(mdfile).readlines()
    mdfile_updated = []
    for line in mdcontent:
        line = replaceLink(line, tutorial_type, mdfile)
        mdfile_updated.append(line)

    with open(mdfile, 'w') as f:
        f.write("".join(mdfile_updated))


def updateLinkLoop(target_dir):
    """ Replace all links in the tutorials of src/assets to fit web application.

    Args:
        target_dir: The path to tutorial dir that is full of md file generated by papermill nbconvert.
    """
    for tutorial_type in ["advanced", "beginner"]:
        for f in os.listdir(os.path.join(target_dir, tutorial_type)):
            if f.endswith('.md'):
                updateLink(os.path.join(target_dir, tutorial_type, f),
                           tutorial_type)


def createJson(target_dir):
    """ Create structure.json

    Args:
        target_dir: The path to tutorial dir that is full of md file generated by papermill nbconvert.
            assume file structure need to be:
    """
    dir_arr = []
    for tutorial_type in ["advanced", "beginner"]:
        dir_obj = {
            "name": tutorial_type,
            "displayName": tutorial_type.capitalize() + ' Tutorials',
            "children": None
        }
        children = []
        sub_tutorial_dir = os.path.join(target_dir, tutorial_type)
        for f in sorted(os.listdir(sub_tutorial_dir)):
            if f.endswith('.md'):
                # open updated markdown file and extract table of content
                title = open(os.path.join(sub_tutorial_dir, f)).readlines()[0]
                title = re.sub(RE_SIDEBAR_TITLE, '', title)
                f_obj = {
                    "name": os.path.join(tutorial_type, f),
                    "displayName": title
                }
                children.append(f_obj)
        dir_obj['children'] = children
        dir_arr.append(dir_obj)

    output_struct = os.path.join(target_dir, 'structure.json')
    # write to json file
    with open(output_struct, 'w') as f:
        f.write(json.dumps(dir_arr))


if __name__ == '__main__':
    # take fastestimator dir path and output dir
    FE_DIR = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    BRANCH = sys.argv[3]

    source_path = os.path.join(FE_DIR, "tutorial")
    output_path = os.path.join(OUTPUT_PATH, "tutorial")

    generateMarkdowns(source_path, output_path)  # create markdown
    updateLinkLoop(output_path)
    createJson(output_path)
