""" FastEstimator API parser. This script parses the fastestimator/fastestimator folder and generates the assets files
    for webpage

    The example parsing case where...
    * repo_dir = "fastestimator"
    * output_dir = "branches/r1.2/"
    * branch = "r1.2"

    (repo)                                  (asset)
    fastestimator/fastestimaor/             branches/r1.2/api/
    ├── estimator.py                        ├── fe/
    ├── network.py                          |   ├── build.md
    ├── pipeline.py                         |   ├── Esimator.md
    ├── backend/                            |   ├── Network.md
    │   ├── abs.py              =>          |   ├── Pipeline.md
    │   ├── argmax.py                       |   └── ....
    │   └── ...                             ├── backend/
    └── ...                                 │   ├── abs.md
                                            │   ├── argmax.md
                                            │   └── ...
                                            └── ...
    Note:
    * The repo to asset follow the folder-to-folder rules whcih means if there is a folder in the repo, you can see the
        correspoding folder in asset with same name.
    * The asset have "fe" folder to accommodate all files in fastestimator/fastesitmator levels
    * For each folder, the parser will read every python file in that folder level and generate markdown file for each
        function and class.
    * The repo to asset doesn't follow file-to-file rules which means one python file might result in mutilple markdown
      file.

"""
import os
import sys
from os import sep
import shutil
import pdb
import inspect
import re
import pydoc
import json


class APIFolder:
    """ class to generate an parsed API folder from an target folder
    it doens't care the subfolder. It only reads the modules in that folder level and pop the markdown files.

    Args:
        save_dir: path to the destination folder (ex: .../api/architecture)
        fe_dir: path to the fastestimator folder (ex: .../fastestimator)
        target_dir: path to the target folder to parse (ex: .../fastestimator/architecture)
    """
    def __init__(self, save_dir, fe_dir, target_dir):

        self.save_dir = save_dir
        self.fe_dir = fe_dir
        self.target_dir = target_dir
        self.func_and_classes = None
        self.file_modules = None

    def _extract_folder_module(self):
        rel_path = os.path.relpath(self.target_dir, self.fe_dir).replace(sep, ".")
        dir_module = pydoc.safeimport(rel_path)

        self.func_and_classes = []
        self.file_modules = []
        # pdb.set_trace()
        for name, member in inspect.getmembers(dir_module):
            if name.startswith("_"):
                continue
            if not member: # ex: EarlyStop
                continue

            source_path = inspect.getfile(member)
            if not source_path.startswith(self.fe_dir): # skip module not from fe ex: cv2
                continue
            if os.path.basename(
                    source_path) == "__init__.py":  # skip folder module
                continue
            if inspect.isfunction(member) or inspect.isclass(member):
                self.func_and_classes.append([name, member])
            elif inspect.ismodule(member):
                self.file_modules.append([name, member])

        # remove file_modules in which any member of func_and_classes is defined
        # ex: >>> from fastestimator.estimator import Estimator
        #     this will include both estimator.py and Estimator but we only want Estimator
        for idx, (name, module) in reversed(list(enumerate(self.file_modules))):
            for _, member in self.func_and_classes:
                if inspect.getmodule(member) == module:
                    self.file_modules.pop(idx)
                    break

    def _build_api_markdown(self):
        for name, obj in self.file_modules:
            APIMarkdown(name=name, obj=obj, save_dir=self.save_dir, fe_dir=self.fe_dir).dump()
        for name, obj in self.func_and_classes:
            APIMarkdown(name=name, obj=obj, save_dir=self.save_dir, fe_dir=self.fe_dir).dump()

    def dump(self):
        self._extract_folder_module()

        if self.file_modules or self.func_and_classes:
            # make sure there is any markdown to be dumped before create the dir (important)
            # otherwise the structure.json will have a folder node without children
            os.makedirs(self.save_dir, exist_ok=True)
            self._build_api_markdown()


class APIMarkdown:
    fe_url = 'https://github.com/fastestimator/fastestimator/blob'

    def __init__(self, name, obj, save_dir, fe_dir):
        self.name = name
        self.obj = obj
        self.save_dir = save_dir
        self.fe_dir = fe_dir

    def get_url(self, obj):
        sourcefile = inspect.getsourcefile(obj)
        sourcelines = inspect.getsourcelines(obj)
        start = sourcelines[1]
        end = start + len(sourcelines[0]) - 1

        url = os.path.join(APIMarkdown.fe_url, branch,
                           os.path.relpath(sourcefile, self.fe_dir),
                           '#L' + str(start) + '-L' + str(end))
        return url

    def get_source_link(self, obj):
        return f'<a class="sourcelink" href={self.get_url(obj)}>View source on Github</a>\n'

    @staticmethod
    def get_type_tag(tag_name):
        return f'<span class="tag">{tag_name}</span>'

    @staticmethod
    def process_module_name(name):
        """ fastestimator -> fe.dataset.data.cifar10
        """
        return re.sub("^fastestimator\.", "fe.", name)

    def file_to_md(self, obj):
        content = []
        content.append(f'# {self.process_module_name(obj.__name__)}')
        content.append(self.get_type_tag("module"))
        content.append("\n\n")
        for name, api_module in inspect.getmembers(obj):
            if inspect.getmodule(api_module) != obj:
                continue
            if name.startswith('_'):
                continue

            if inspect.isclass(api_module):
                content.extend(self.class_to_md(api_module, True))
            elif inspect.isfunction(api_module):
                content.extend(self.func_to_md(api_module, True))

        return content

    def func_to_md(self, obj, toprule):
        content = []
        if toprule:
            content.append("---\n\n")
        content.append(f'## {obj.__name__}')
        content.append(self.get_type_tag('function'))
        content.append(self.get_source_link(obj))
        content.append("```python\n")
        content.append(obj.__name__)
        signature = str(inspect.signature(obj))
        content.append(self.prettify_signature(signature) + '\n')
        content.append('```\n')
        docstr = DocString(obj.__doc__)
        content.append(docstr.format(level=3))

        return content

    def class_to_md(self, obj, toprule):
        try: # some class raise error when running inspect.signature(self.obj) such as fe.EarlyStop
            old_sig = inspect.signature(obj.__init__)
            params = [x for x in old_sig.parameters.values()]
            # create new signature without self argument
            signature = str(old_sig.replace(parameters=params[1:]))

        except ValueError:
            return []

        content = []
        if toprule:
            content.append("---\n\n")
        content.append(f'## {obj.__name__}')
        content.append(self.get_type_tag("class"))
        content.append(self.get_source_link(obj))
        content.append("```python\n")
        content.append(obj.__name__)

        content.append(self.prettify_signature(signature) + "\n")
        content.append('```\n')

        docstr = DocString(obj.__doc__)
        content.append(docstr.format(level=3))
        content.extend(self.get_class_method(obj))

        return content


    def get_class_method(self, obj):
        """It extracts the functions which are functions of the current class that is being

        Returns:
            [list]: It returns the markdown string with object signature appended in the list
        """
        content = []
        funcs = inspect.getmembers(obj, inspect.isfunction)
        for name, func in funcs:
            if func.__qualname__.split('.')[0] == obj.__qualname__:
                if name.startswith("_"):
                    continue
                if not func.__doc__: # otherwise trace will show on_epoch method ...
                    continue

                content.append("---\n\n")
                content.append('### ' + name)
                content.append(self.get_type_tag(f"method of {obj.__name__}"))
                content.append(self.get_source_link(func))
                content.append("```python\n")
                content.append(name)
                signature = str(inspect.signature(func))
                content.append(self.prettify_signature(signature) + "\n")
                content.append('```\n')
                docstr = DocString(func.__doc__)
                content.append(docstr.format(level=4))

        return content


    def dump(self):
        if inspect.isclass(self.obj):
            content = self.class_to_md(self.obj, False)
        elif inspect.isfunction(self.obj):
            content = self.func_to_md(self.obj, False)
        else:
            content = self.file_to_md(self.obj)

        save_path = os.path.join(self.save_dir, self.name + ".md")
        with open(save_path, "w") as f:
            f.write("".join(content))

    @staticmethod
    def prettify_signature(inp: str):
        out = inp
        out = re.sub(r'^\((.+)\)', r'(\n\t\1\n)', out)
        out = re.sub(r' ->', r'\n->', out)
        out = re.sub(r'(, )([\w\d]+:)', r',\n\t\2', out)
        out = re.sub(r':', r': ', out)

        return out

class DocString:
    def __init__(self, string):
        self.string = string
        self.intro = ''
        self.args = ''
        self.raises = ''
        self.returns = ''

    def format(self, level):
        self._parse()

        content = []
        content.append(self.intro + "\n\n")
        if self.args:
            content.append(self._format_args(level))

        if self.returns:
            content.append(self._format_returns(level))

        if self.raises:
            content.append(self._format_raises(level))

        return "".join(content)

    def _format_returns(self, level):
        header = f"<h{level}>Returns:</h{level}>"

        #  `xxxx` cannot render properly inside html tag
        content = re.sub(r"`([^`]+)`", r"<code>\1</code>", self.returns)
        content = f'<ul class="return-block"><li>{content}</li></ul>'

        return header + "\n\n" + content + "\n\n"

    def _format_raises(self, level):
        header = f"<h{level}>Raises:</h{level}>"
        content = re.sub(r"( +)([^\s]+): ", r"\n* **\2**: ", self.raises)

        return header + "\n\n" + content + "\n\n"


    def _format_args(self, level):
        header = f"<h{level}>Args:</h{level}>"
        content = re.sub(r"( {3,})([^\s]+): ", r"<newline>* **\2**: ", self.args)
        content = re.sub(r"[\n ]{2,}", r" ", content)
        content = re.sub(r"<newline>", r"\n", content)

        return header + "\n\n" + content + "\n\n"

    def _parse(self):
        self.intro = ''
        self.args = ''
        self.raises = ''
        self.returns = ''

        if not self.string:
            return

        docstr = inspect.cleandoc(self.string)
        docstr = DocString.replace_escape_char(docstr)
        docstr = DocString.unwrap_http_url(docstr)
        pattern = re.compile(r"(Args|Returns|Raises):\n")
        matches = [match for match in pattern.finditer(docstr)]

        if not matches:
            self.intro = docstr
        else:
            self.intro = docstr[:matches[0].start()-1]

            for idx in range(len(matches)):
                if idx + 1 < len(matches): # not the last one
                    part = docstr[matches[idx].end(): matches[idx + 1].start()]
                else: # last one
                    part = docstr[matches[idx].end():]

                if matches[idx].group() == "Args:\n":
                    self.args = part
                elif matches[idx].group() == "Returns:\n":
                    self.returns = part
                elif matches[idx].group() == "Raises:\n":
                    self.raises = part

    @staticmethod
    def replace_escape_char(inpstr, startpos=0):
        re_plot = r'```\s*(plot|python)([^```]+)```'
        pattern = re.compile(re_plot)
        res = pattern.search(inpstr, startpos)
        if res is not None:
            startidx = res.start()
            endidx = res.end()
            if startpos != 0:
                output = inpstr[0:startpos] + inpstr[startpos:startidx].replace(
                    '<', '&lt;').replace('>', '&gt;')
            else:
                output = inpstr[startpos:startidx].replace('<', '&lt;').replace(
                    '>', '&gt;')
            output = '{}{}\n{}'.format(output, res.group(0), inpstr[endidx + 1:])
            return DocString.replace_escape_char(output, startpos=endidx)
        else:
            return inpstr

    @staticmethod
    def remove_newline(match):
        return re.sub(r"\n", "", match.group(0))

    @staticmethod
    def unwrap_http_url(inpstr):
        """ make url from multiple line back to one line
            ex:
                'https://papers.nips.cc/paper/9070-error
                -correcting-output-codes'
                =>
                'https://papers.nips.cc/paper/9070-error-correcting-output-codes'
        """
        url_pattern = r"'(http|https)://[^']*'"
        return re.sub(url_pattern, DocString.remove_newline, inpstr)


def generatedocs(repo_dir, save_dir, branch):
    """This function loop through files and sub-directories in project directory in top down approach and get python code
    files to extract markdowns. It also prepares path to save markdown file for corresponding python file.

    Args:
        path[str]: Path to the project which markdown string are extracted

    Returns:
        [str]: Returns absolute path to the generated markdown directory
    """
    #insert project path to system path to later detect the modules in project
    sys.path.insert(0, repo_dir)

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    fe_path = os.path.abspath((os.path.join(repo_dir, 'fastestimator')))

    # directories that needs to be excluded
    exclude = set(['test', '__pycache__'])
    for path, dirs, _ in os.walk(fe_path, topdown=True):
        if path == fe_path:
            save_path = os.path.join(save_dir, 'fe')
        else:
            save_path = os.path.join(save_dir, os.path.relpath(path, fe_path))

        os.makedirs(save_path, exist_ok=True)
        dirs[:] = [d for d in dirs if d not in exclude]
        api_folder = APIFolder(save_dir=save_path,
                               target_dir=path,
                               fe_dir=repo_dir)
        api_folder.dump()


    return save_dir


def generate_json(path):
    """This function generates JSON file that represents the file structure of the markdown files. JSON file is rendered along
    with the markdown files to create API webpages.

    Args:
        path[str]: Path to the generated markdown files

    Returns:
        [list]: list which contains file structure of the markdown files
    """
    fe_path = path  #keep a copy of the path to later use it in recursive calls

    def generate_dict(path):
        rel_path = os.path.relpath(path, fe_path)
        json_dict = {"name": rel_path, "displayName":None}

        # directory
        if os.path.isdir(path):
            # check if the path is under fe
            if rel_path.split(sep)[0] == "fe":
                display_name = rel_path.replace(sep, ".")
            else:
                display_name = "fe." + rel_path.replace(sep, ".")

            # list of children from directories
            children_dirs = [
                generate_dict(os.path.join(path, x)) for x in os.listdir(path)
                if os.path.isdir(os.path.join(path, x))
            ]

            # list of children from files
            children_files = [
                generate_dict(os.path.join(path, x)) for x in os.listdir(path)
                if os.path.isfile(os.path.join(path, x))
            ]

            # the order of children list is "directories then files"
            children = sorted(children_dirs, key=lambda x: x["displayName"])
            children.extend(sorted(children_files, key=lambda x: x["displayName"]))

            json_dict["children"] = children
        # file
        else:
            display_name = os.path.splitext(os.path.basename(path))[0]

        json_dict["displayName"] = display_name

        return json_dict

    if os.path.isdir(fe_path):
        json_list = [
            generate_dict(os.path.join(fe_path, x)) for x in os.listdir(fe_path)
        ]
        json_list = sorted(json_list, key=lambda x: x['displayName'])
        return json_list


if __name__ == "__main__":
    repo_dir = sys.argv[1]
    output_dir = sys.argv[2]
    branch = sys.argv[3]

    save_dir = os.path.join(output_dir, "api")
    json_path = os.path.join(save_dir, "structure.json")

    generatedocs(repo_dir, save_dir, branch)
    struct_json = generate_json(save_dir)

    with open(json_path, 'w') as f:
        json.dump(struct_json, f, indent=4)
