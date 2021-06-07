""" FastEstimator tutorial parser. This script parses the FE tutorial and generates the assets files for webpage

    The example parsing case where...
    * FE_DIR = "fastestimator"
    * OUTPUT_PATH = "branches/r1.2/"

    (repo)                                  (asset)
    fastestimator/tutorial/                 branches/r1.2/tutorial
    ├── advanced/                           ├── advanced/
    │   ├── t01_dataset.ipynb               │   ├── t01_dataset.md
    │   ├── t02_pipeline.ipynb              │   ├── t02_pipeline.md
    │   └── ...                             │   └── ...
    ├── beginner/                           ├── beginner/
    │   ├── t01_getting_started.ipynb   =>  │   ├── t01_getting_started.md
    │   ├── t02_dataset.ipynb               |   ├── t01_getting_started_files/
    │   └── ...                             |   |   └─ t01_getting_started_19_1.png
    └── resources                           |   ├── t02_dataset.md
        ├── t01_api.png                     |   └── ...
        └── ...                             ├── resources/
                                            |   ├── t01_api.png
                                            |   └── ...
                                            └── structure.json

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
from shutil import copy

T_NAME = {"repo": "tutorial", "asset": "tutorial", "route": "tutorials"}

A_NAME = {"repo": "apphub", "asset": "example", "route": "examples"}


def generate_md(source_tutorial, output_tutorial):
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

    # copy resources dir
    source_resources = os.path.join(source_tutorial, "resources")
    output_resources = os.path.join(output_tutorial, "resources")
    shutil.copytree(source_resources, output_resources)


def replace_link_to_tutorial(match):
    """
    pattern = f'^(.+)\.ipynb$'

    ex: ../beginner/t02_dataset
     -> tutorial/r1.2/beginner/t02_dataset
    """

    return f'{T_NAME["route"]}/{BRANCH}/{match.group(1)}'


def replace_link_to_tutorial_tag(match):
    """
    pattern = f'^(.+)\.ipynb#(.+)$'

    ex: ../beginner/t02_dataset#t02Apphub
     -> tutorial/r1.2/beginner/t02_dataset#t02Apphub
    """

    return f'{T_NAME["route"]}/{BRANCH}/{match.group(1)}#{match.group(2)}'


def replace_link_to_apphub(match):
    """
    pattern = f'^\.\.\/{A_NAME["repo"]}\/(.+)\.ipynb$'

    ex: ../../apphub/adversarial_training/fgsm/fgsm.ipynb
        examples/r1.2/adversarial_training/fgsm/fgsm.ipynb
    """
    return f'{A_NAME["route"]}/{BRANCH}/{match.group(1)}'


def replace_link_to_apphub_tag(match):
    """
    pattern = f'^\.\.\/{A_NAME["repo"]}\/(.+)\.ipynb#(.+)$'

    ex: ../../apphub/adversarial_training/fgsm/fgsm.ipynb#model-testing
     -> examples/r1.2/adversarial_training/fgsm/fgsm.ipynb#model-testing
    """
    return f'{A_NAME["route"]}/{BRANCH}/{match.group(1)}#{match.group(2)}'


def update_link_to_page(link, rel_path):
    """
    tutorial to apphub
    ex: ../../apphub/adversarial_training/fgsm/fgsm.ipynb
     -> examples/r1.2/adversarial_training/fgsm/fgsm.ipynb

    tutorial to tutorial
    ex: ../beginner/t02_dataset
     -> tutorial/r1.2/beginner/t02_dataset
    """

    rel_path_link = os.path.normpath(os.path.join(rel_path, link))

    apphub_pattern = f'^\.\.\/{A_NAME["repo"]}\/(.+)\.ipynb$'
    apphub_pattern_tag = f'^\.\.\/{A_NAME["repo"]}\/(.+)\.ipynb#(.+)$'

    tutorial_pattern = f'^(.+)\.ipynb$'
    tutorial_pattern_tag = f'^(.+)\.ipynb#(.+)$'

    if rel_path_link.startswith(f'../{A_NAME["repo"]}'):  # to apphub
        if re.search(apphub_pattern, rel_path_link):
            link = re.sub(apphub_pattern, replace_link_to_apphub,
                          rel_path_link)
        elif re.search(apphub_pattern_tag, rel_path_link):
            link = re.sub(apphub_pattern_tag, replace_link_to_apphub_tag,
                          rel_path_link)

    elif rel_path_link.startswith(f'../'):
        raise RuntimeError(
            "The page link need to point to either tutorial or apphub")
    else:  # to tutorial
        if re.search(tutorial_pattern, rel_path_link):
            link = re.sub(tutorial_pattern, replace_link_to_tutorial,
                          rel_path_link)
        elif re.search(tutorial_pattern_tag, rel_path_link):
            link = re.sub(tutorial_pattern_tag, replace_link_to_tutorial_tag,
                          rel_path_link)

    return link


def update_link_to_asset(link, rel_path):
    """to repo asset file
    ex: ../resources/t01_api.png
     -> assets/branches/r1.2/tutorial/resources/t01_api.png
    """
    prefix = f"assets/branches/{BRANCH}"
    rel_path_link = os.path.normpath(
        os.path.join(T_NAME["asset"], rel_path, link))

    return os.path.join(prefix, rel_path_link)


def update_link_pure_tag(link, rel_path, fname):
    """to same apphub with tag
    ex: #t02Apphub
     -> tutorial/r1.2/beginner/t02_dataset#t02Apphub
    """
    fname, _ = os.path.splitext(fname)
    link = os.path.join(T_NAME["route"], BRANCH, rel_path, fname, link)
    return link


def update_link(link, rel_path, fname):
    link = link.strip()
    url_pattern = r'^https?:\/\/|^www.'
    nb_pattern = r'\.ipynb$|\.ipynb#(.)+$'
    tag_pattern = r'^#[^\/]+$'

    if re.search(url_pattern, link):  # if the link is url
        pass
    elif re.search(nb_pattern, link):  # if it points to nb file
        link = update_link_to_page(link, rel_path)

    elif re.search(tag_pattern, link):  # if it points to pure hashtag
        link = update_link_pure_tag(link, rel_path, fname)

    else:
        link = update_link_to_asset(link, rel_path)

    return link


def update_line(line, rel_path, fname):
    link_pattern1 = r'\[([^\[]+)\]\(([^\)]+)\)'
    link_pattern2 = r'src=[\"\']([^\"\']+)[\"\']'
    link_pattern3 = r'href=[\"\']([^\"\']+)[\"\']'
    if re.search(link_pattern1, line):
        line = re.sub(
            link_pattern1, lambda match:
            f'[{match.group(1)}]({update_link(match.group(2), rel_path, fname)})',
            line)
    elif re.search(link_pattern2, line):
        # pdb.set_trace()
        line = re.sub(
            link_pattern2, lambda match:
            f'src={update_link(match.group(1), rel_path, fname)}', line)

    elif re.search(link_pattern3, line):
        line = re.sub(
            link_pattern3, lambda match:
            f'href={update_link(match.group(1), rel_path, fname)}', line)

    return line


def update_md(target_dir):
    """ Replace all links in the tutorials of src/assets to fit web application.

    Args:
        target_dir: The path to tutorial dir that is full of md file generated by papermill nbconvert.
    """
    for path, dirs, files in os.walk(target_dir):
        rel_path = os.path.relpath(path, target_dir)
        for f in files:
            if f.endswith('.md'):
                md_path = os.path.join(path, f)
                mdcontent = open(md_path).readlines()
                mdfile_updated = []
                for line in mdcontent:
                    line = update_line(line, rel_path, f)
                    mdfile_updated.append(line)

                with open(md_path, 'w') as f:
                    f.write("".join(mdfile_updated))


def create_json(target_dir):
    """ Create structure.json

    Args:
        target_dir: The path to tutorial dir that is full of md file generated by papermill nbconvert.
            assume file structure need to be:
    """
    dir_arr = []
    for tutorial_type in ["beginner", "advanced"]:
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
                title = re.sub(r'[^A-Za-z0-9:!,$%. ()]+', '',
                               title)  # remove illegal characters
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
    FE_DIR = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    BRANCH = sys.argv[3]

    source_path = os.path.join(FE_DIR, "tutorial")
    output_path = os.path.join(OUTPUT_PATH, "tutorial")

    generate_md(source_path, output_path)
    update_md(output_path)
    create_json(output_path)
