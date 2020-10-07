import os
import sys
from os import sep
import shutil
import pdb
import inspect
import re
import pydoc
import json

titles = ['Args', 'Raises', 'Returns']
fe_url = 'https://github.com/fastestimator/fastestimator/blob'
html_char_regex = r'([\<\>])'
args_regex = r'(Args|Returns|Raises):\n'
re_url = r'(?:(?:http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.(?:[a-zA-Z]){2,6}(?:[a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


def isDoc(obj):
    doc = obj.__doc__
    if doc == '' or doc == None:
        return True
    return False

class APIFolder:
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
            if not member: # EarlyStop
                continue
            if os.path.basename( # skip folder module
                    inspect.getsourcefile(member)) == "__init__.py":
                continue
            # print(f"name:{name} member:{member}")
            # pdb.set_trace()
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
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._extract_folder_module()
        self._build_api_markdown()


class APIMarkdown:
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

        url = os.path.join(fe_url, branch, os.path.relpath(sourcefile, self.fe_dir),
                           '#L' + str(start) + '-L' + str(end))

        return url

    def get_source_link(self, obj):
        return f'<a class="sourcelink" href={self.get_url(obj)}>View source on Github</a>\n'

    @staticmethod
    def get_type_tag(tag_name):
        return f'<span class="tag">{tag_name}</span>'

    def file_to_md(self, obj):
        content = []
        content.append(f'# {obj.__name__}')
        content.append(self.get_type_tag("module"))
        content.append("\n---\n")
        for name, api_module in inspect.getmembers(obj):
            if inspect.getmodule(api_module) != obj:
                continue
            if name.startswith('_'):
                continue

            if inspect.isclass(api_module):
                content.extend(self.class_to_md(api_module))
            elif inspect.isfunction(api_module):
                content.extend(self.func_to_md(api_module))

        return content

    def func_to_md(self, obj):
        content = []
        content.append(f'## {obj.__name__}')
        content.append(self.get_type_tag('function'))
        content.append(self.get_source_link(obj))
        content.append("```python\n")
        content.append(obj.__name__)
        signature = str(inspect.signature(obj))
        content.append(self.prettify_signature(signature) + '\n')
        content.append('```\n')
        content.append(self.format_docstring(inspect.getdoc(obj), level=3))
        content.append('\n\n')

        return content


    def class_to_md(self, obj):
        try: # some class raise error when running inspect.signature(self.obj) such as fe.EarlyStop
            signature = str(inspect.signature(obj))
        except ValueError:
            return []

        content = []
        content.append(f'## {obj.__name__}')
        content.append(self.get_type_tag("class"))
        content.append(self.get_source_link(obj))
        content.append("```python\n")
        content.append(obj.__name__)

        content.append(self.prettify_signature(signature) + "\n")
        content.append('```\n')
        content.append(self.format_docstring(obj.__doc__, level=3))
        content.append('\n\n')
        content.extend(self.get_class_method(obj))
        content.append('\n\n')


        return content


    def get_class_method(self, obj):
        """It extracts the functions which are functions of the current class that is being

        Returns:
            [list]: It returns the markdown string with object signature appended in the list
        """
        content = []
        funcs = inspect.getmembers(obj, inspect.isfunction)
        for f in funcs:
            if f[1].__qualname__.split('.')[0] == obj.__qualname__:
                if not f[0].startswith("_") and not isDoc(f[1]):
                    content.append('### ' + f[0])
                    content.append(self.get_type_tag(f"method of {obj.__name__}"))
                    content.append(self.get_source_link(f[1]))
                    content.append("```python\n")
                    content.append(f[0])
                    signature = str(inspect.signature(f[1]))
                    content.append(APIMarkdown.prettify_signature(signature) + "\n")
                    content.append('```\n')
                    content.append(APIMarkdown.format_docstring(f[1].__doc__, level=4))
                    content.append('\n\n')

        return content


    def dump(self):
        if inspect.isclass(self.obj):
            content = self.class_to_md(self.obj)
        elif inspect.isfunction(self.obj):
            content = self.func_to_md(self.obj)
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

    @staticmethod
    def format_docstring(docstr, level):
        """It format the docstring in markdown form and append it to the list

        Args:
            docstr[str]: Input docstring that needs to be formated

        Returns:
            [str]: markdown string converted from docstring
        """
        docstr_output = ''
        if docstr != None:
            docstr_output = inspect.cleandoc(docstr)
            docstr_output = APIMarkdown.replaceEscapeChar(docstr_output)
            args = re.search(args_regex, docstr_output)
            if args != None:
                pos = args.start(0)
                doc_arg = docstr_output[pos:]
                docstr_output = docstr_output[:pos - 1]
                urls = re.findall(re_url, doc_arg)
                new_docstr = doc_arg.split('\n')
                res = []
                if len(new_docstr) != 0:
                    for idx in range(len(new_docstr)):
                        if ':' in new_docstr[idx] and new_docstr[idx].strip() not in urls:
                            elements = new_docstr[idx].split(':')
                            if elements[0].strip() in titles:
                                res.append(
                                    f"\n\n<h{level}>{elements[0].strip()}:</h{level}>\n"
                                )
                            else:
                                res.append('\n')
                                param = elements[0].strip()
                                if param[0] in ['*']:
                                    param = ' ' + param
                                else:
                                    param = '* **' + param + '**'
                                res.append(param)
                                res.append(' : ')
                                for i in range(1, len(elements)):
                                    res.append(elements[i])
                        else:
                            res.append(new_docstr[idx])
                docstr_output = docstr_output + ''.join(res)
            return docstr_output
        return docstr_output

    @staticmethod
    def replaceEscapeChar(input, startpos=0):
        re_plot = r'```\s*(plot|python)([^```]+)```'
        pattern = re.compile(re_plot)
        res = pattern.search(input, startpos)
        if res is not None:
            startidx = res.start()
            endidx = res.end()
            if startpos != 0:
                output = input[0:startpos] + input[startpos:startidx].replace(
                    '<', '&lt;').replace('>', '&gt;')
            else:
                output = input[startpos:startidx].replace('<', '&lt;').replace(
                    '>', '&gt;')
            output = '{}{}\n{}'.format(output, res.group(0), input[endidx + 1:])
            return APIMarkdown.replaceEscapeChar(output, startpos=endidx)
        else:
            return input


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

    fe_path = os.path.abspath((os.path.join(repo_dir, 'fastestimator')))

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    # directories that needs to be excluded
    exclude = set(['test', '__pycache__'])
    for path, dirs, _ in os.walk(fe_path, topdown=True):
        if path == fe_path:
            save_path = os.path.join(save_dir, 'fe')
        else:
            save_path = os.path.join(save_dir, os.path.relpath(path, fe_path))

        os.makedirs(save_path, exist_ok=True)
        dirs[:] = [d for d in dirs if d not in exclude]

        # if path == "/home/geez219/python_project/fastestimator/fastestimator/dataset":
        api_folder = APIFolder(save_dir=save_path,
                                target_dir=path,
                                fe_dir=repo_dir)
        api_folder.dump()
        # pdb.set_trace()
        # print()

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
    branch = sys.argv[2]
    output_dir = sys.argv[3]

    save_dir = os.path.join(output_dir, "api")
    json_path = os.path.join(save_dir, "structure.json")
    generatedocs(repo_dir, save_dir, branch)
    struct_json = generate_json(save_dir)

    with open(json_path, 'w') as f:
        json.dump(struct_json, f, indent=4)


    # ## experiment
    # branch = "master"
    # fe_path = "/home/geez219/python_project/fastestimator/fastestimator"
    # sys.path.insert(0, fe_path)
    # name = "Estimator"
    # save_dir = "Estimator"
    # f_rel_path = "estimator.py"
    # os.makedirs(save_dir, exist_ok=True)
    # module = pydoc.safeimport(inspect.getmodulename("estimator.py"))
    # api_md = APIMarkdown(name=name,
    #                      module=module,
    #                      save_dir=save_dir,
    #                      f_rel_path=f_rel_path)
    # api_md.dump()
