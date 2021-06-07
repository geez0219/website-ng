""" FastEstimator apphub parser. This script parses the FE apphub and generates the assets files for webpage

    The example parsing case where...
    * FE_DIR = "fastestimator"
    * OUTPUT_PATH = "branches/r1.2/"

    (repo)                                     (asset)
    fastestimator/apphub/                      branches/r1.2/example/
    ├── image_generation/                      ├── image_generation/
    │   ├── cvae/                              │   ├── cvae/
    │   |   ├── cvae.ipynb                     │   |   ├── cvae.md
    |   |   └── VAE_complete.png               |   |   ├── cvae_files/
    |   |── pggan/                             |   |   |   └──  cvae_11_0.png
    |   |   ├── pggan.ipynb              =>    |   |   └── VAE_complete.png
    |   |   └── Figure/                        |   |── pggan/
    |   |       ├── pggan_1024x1024.png        |   |   ├── pggan.md
    |   |       └── ...                        |   |   ├── pggan_files/
    |   └── ...                                |   |   |   └── pggan_10_0.png
    └── ...                                    |   |   └─── Figure/
                                               |   |        ├── pggan_1024x1024.png
                                               |   |        └── ...
                                               |   └── ...
                                               ├── ...
                                               ├── overview.md
                                               └── structure.json
    Note:
    * The difference between this parsing script with tutorial_parser.py is very small small. In the future we could
      merge these two scripts together.
    * One of the difference is it reads the apphub/README.md and generate the overview.md and structure.json

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


def generate_md(source_apphub, output_apphub):
    """ Parse apphub from repo and generate asset of apphub for webpage.
    Args:
        source_apphub: tutorial source (ex: fastestimator/tutorial)
        output_apphub: parsed tutorial destination (ex: r1.1/tutorial)
    """

    if os.path.exists(output_apphub):
        shutil.rmtree(output_apphub)

    for path, dirs, files in os.walk(source_apphub):
        # not walk into the dir start with "_" and ".",
        dirs[:] = [d for d in dirs if d[0] not in ["_", "."]]
        output_dir = os.path.join(output_apphub,
                                  os.path.relpath(path, source_apphub))
        os.makedirs(output_dir, exist_ok=True)

        case_dir = False  # is the current dir has ipynb?
        for f in files:
            if f.endswith('.ipynb'):
                filepath = os.path.join(path, f)
                # invoke subprocess to run nbconvert command on notebook files
                subprocess.run([
                    'jupyter', 'nbconvert', '--to', 'markdown', filepath,
                    '--output-dir', output_dir
                ])
                case_dir = True

        if case_dir:
            # copy all resources that .ipynb might need
            for f in files:
                if not f.endswith('.ipynb') and not f.endswith(".py"):
                    filepath = os.path.join(path, f)
                    shutil.copy(filepath, output_dir)

            for d in dirs:
                dirpath = os.path.join(path, d)
                newdir = os.path.join(output_dir, d)
                shutil.copytree(dirpath, newdir)


def replace_link_to_tutorial(match):
    """
    pattern = f'^\.\.\/{T_NAME["repo"]}\/(.+)\.ipynb$'

    ex: ../../tutorial/beginner/t01_getting_started.ipynb
     -> tutorials/r1.2/beginner/t01_getting_started
    """

    return f'{T_NAME["route"]}/{BRANCH}/{match.group(1)}'


def replace_link_to_tutorial_tag(match):
    """
    pattern = f'^\.\.\/{T_NAME["repo"]}\/(.+)\.ipynb#(.+)$'

    ex: ../../tutorial/beginner/t01_getting_started.ipynb#hello
     -> tutorials/r1.2/beginner/t01_getting_started#hello
    """

    return f'{T_NAME["route"]}/{BRANCH}/{match.group(1)}#{match.group(2)}'


def replace_link_to_apphub(match):
    """
    pattern = f'^(.+)\.ipynb$'

    ex: ../../image_classification/cifar10_fast/cifar10_fast.ipynb
     -> examples/r1.2/image_classification/cifar10_fast/cifar10_fast
    """
    return f'{A_NAME["route"]}/{BRANCH}/{match.group(1)}'


def replace_link_to_apphub_tag(match):
    """
    pattern = f'^(.+)\.ipynb#(.+)$'

    ex: ../../image_classification/cifar10_fast/cifar10_fast.ipynb#hello
     -> examples/r1.2/image_classification/cifar10_fast/cifar10_fast#hello
    """
    return f'{A_NAME["route"]}/{BRANCH}/{match.group(1)}#{match.group(2)}'


def update_link_to_page(link, rel_path):
    """
    apphub to apphub
    ex: ../../image_classification/cifar10_fast/cifar10_fast.ipynb
     -> examples/r1.2/image_classification/cifar10_fast/cifar10_fast

    apphub to tutorial
    ex: ../../tutorial/beginner/t01_getting_started.ipynb
     -> ./tutorials/r1.2/beginner/t01_getting_started
    """

    rel_path_link = os.path.normpath(os.path.join(rel_path, link))

    tutorial_pattern = f'^\.\.\/{T_NAME["repo"]}\/(.+)\.ipynb$'
    tutorial_pattern_tag = f'^\.\.\/{T_NAME["repo"]}\/(.+)\.ipynb#(.+)$'
    apphub_pattern = f'^(.+)\.ipynb$'
    apphub_pattern_tag = f'^(.+)\.ipynb#(.+)$'

    if rel_path_link.startswith(f'../{T_NAME["repo"]}'): # to tutorial
        if re.search(tutorial_pattern, rel_path_link):
            link = re.sub(tutorial_pattern, replace_link_to_tutorial,
                          rel_path_link)
        elif re.search(tutorial_pattern_tag, rel_path_link):
            link = re.sub(tutorial_pattern_tag, replace_link_to_tutorial_tag,
                          rel_path_link)
    elif rel_path_link.startswith(f'../'):
        raise RuntimeError(
            "The page link need to point to either tutorial or apphub")
    else:  # to apphub
        if re.search(apphub_pattern, rel_path_link):
            link = re.sub(apphub_pattern, replace_link_to_apphub,
                          rel_path_link)
        elif re.search(apphub_pattern_tag, rel_path_link):
            link = re.sub(apphub_pattern_tag, replace_link_to_apphub_tag,
                          rel_path_link)

    return link



def update_link_to_asset_api(link, rel_path):
    """to repo asset files or api folder
    to api
    ex: ../../fastestimator/architecture
    ->  https://github.com/fastestimator/fastestimator/tree/r1.2/fastestimator/architecture

    to asset
    ex: ./Figure/pggan_1024x1024.png
    ->  assets/branches/r1.2/example/image_generation/pggan/Figure/pggan_1024x1024.png
    """

    rel_path_link = os.path.normpath(
        os.path.join(A_NAME["asset"], rel_path, link))
    if rel_path_link.startswith(f'fastestimator'):
        # link to API
        # TODO: link to fastestimator API webpage
        link = f"https://github.com/fastestimator/fastestimator/tree/{BRANCH}/{rel_path_link}"
    else:
        # link to asset
        link = f"assets/branches/{BRANCH}/{rel_path_link}"

    return link


def update_link_pure_tag(link, rel_path, fname):
    """to same apphub with tag
    ex: #hello
     -> examples/r1.2/image_classification/cifar10_fast/cifar10_fast#hello
    """
    fname, _ = os.path.splitext(fname)
    link = os.path.join(A_NAME["route"], BRANCH, rel_path, fname, link)
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
        link = update_link_to_asset_api(link, rel_path)

    return link


def update_line(line, rel_path, fname):
    link_pattern1 = r'\[(.+)\]\((.+)\)'
    link_pattern2 = r'src=[\"\']([^ ]+)[\"\']'
    link_pattern3 = r'href=[\"\']([^ ]+)[\"\']'
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
    """ Replace all links in the examples of src/assets to fit web application.

    Args:
        target_dir: The path to apphub dir that is full of md file generated by papermill nbconvert.
    """
    for path, dirs, files in os.walk(target_dir):
        rel_path = os.path.relpath(path, target_dir)
        for f in files:
            if f.endswith('.md') and f != "overview.md":
                md_path = os.path.join(path, f)
                mdcontent = open(md_path).readlines()
                mdfile_updated = []
                for line in mdcontent:
                    line = update_line(line, rel_path, f)
                    mdfile_updated.append(line)

                with open(md_path, 'w') as f:
                    f.write("".join(mdfile_updated))


def create_json(output_path, apphub_path):
    """Generate overview.md and structure.json by parsing the `apphub_path`/README.md

    README.md in apphub has following structure:
        ...
        ## Table of Contents:
        ### Natural Language Processing (NLP)
        ...
        ## Contributions
        ...

    * The part from starting line to "## Table of Contents" called <overview>
    * The part from "## Table of Contents" to the line before "## Contributions" called <toc>
    * This function parses the README.md, generates overview.md, and generate structure.json from parsing <toc>
    """
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

    # write <overview> to the file
    with open(apphub_toc_path, 'w') as f:
        f.write("".join(overview))

    # parse the <toc> to create JSON
    struct = []
    category = None
    for line in toc[1:]:
        if line.startswith("###"):
            if category:  # record the previous category dict
                category["children"].sort(key=lambda x: x["displayName"])
                struct.append(category)

            match = re.search(r'### (.+)', line)
            if match:
                category_dname = match.group(1)
            else:
                raise RuntimeError(
                    f"cannot extract display name of an apphub category from line:\n{line}"
                )

            category = {
                "name": None,
                "displayName": category_dname,
                "children": []
            }

        elif line.startswith("*"):
            assert category is not None
            match = re.search(r'\*\*(.+):\*\*', line)
            match2 = re.search(r'\*\*(.+)\*\*:', line)
            if match:
                example_dname = match.group(1)
            elif match2:
                example_dname = match2.group(1)
            else:
                raise RuntimeError(
                    f"cannot extract display name of an apphub example from line:\n{line}"
                )

            match = re.search(
                r'\[notebook\]\(https://github.com/fastestimator/fastestimator/blob/[^/]+/apphub/(.+).ipynb\)',
                line)
            if match:
                example_name = match.group(1) + ".md"
                category_name = example_name.split("/")[0]
            else:
                raise RuntimeError(
                    f"cannot extract name of an apphub example from line:\n{line}"
                )

            if category["name"] and category["name"] != category_name:
                raise RuntimeError(
                    "example from same category should be under same category dir (apphub/<category_dir>)"
                )

            category["name"] = category_name
            category["children"].append({
                "name": example_name,
                "displayName": example_dname
            })

    # sort the category
    struct.sort(key=lambda x: x["displayName"])

    struct.insert(0, {"name": "overview.md", "displayName": "Overview"})

    # write to json file
    json_path = os.path.join(output_path, 'structure.json')
    with open(json_path, 'w') as f:
        f.write(json.dumps(struct))


if __name__ == '__main__':
    FE_DIR = sys.argv[1]
    OUTPUT_PATH = sys.argv[2]
    BRANCH = sys.argv[3]

    source_path = os.path.join(FE_DIR, A_NAME["repo"])
    output_path = os.path.join(OUTPUT_PATH, A_NAME["asset"])

    generate_md(source_path, output_path)
    update_md(output_path)
    create_json(output_path, source_path)
