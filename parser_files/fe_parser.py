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
sourcelines_dict = {}
html_char_regex = r'([\<\>])'
args_regex = r'(Args|Returns|Raises):\n'
re_url = r'(?:(?:http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.(?:[a-zA-Z]){2,6}(?:[a-zA-Z0-9\.\&\/\?\:@\-_=#])*'


def isDoc(obj):
    doc = obj.__doc__
    if doc == '' or doc == None:
        return True
    return False


def add_source_lines(module, mod_name, f_rel_path, save_dir):
    sourcelines = inspect.getsourcelines(module)
    start = sourcelines[1]
    end = start + len(sourcelines[0]) - 1

    # use dumped markdown absolute path as key
    # /home/geez219/angular_project/website-ng2/parser_files/master/api/dataset/data/cifar10.load_data.md
    key = os.path.abspath(os.path.join(save_dir, mod_name) + ".md")

    # cifar10.load_data -> cifar10/load_data
    mod_name = mod_name.replace(".", sep)

    sourcelines_dict[key] = os.path.join(
        fe_url, branch, 'fastestimator', f_rel_path, '#L' + str(start) + '-L' + str(end))

class APIFolder:
    def __init__(self, save_dir, fe_dir, target_dir):
        self.save_dir = save_dir
        self.fe_dir = fe_dir
        self.target_dir = target_dir
        self.module_infos = None

    def _extract_api_module(self):
        self.module_infos = []
        for f in os.listdir(self.target_dir):
            if not f.endswith(".py"):
                continue
            f_rel_path = os.path.relpath(
                os.path.join(self.target_dir, f),
                self.fe_dir,
            )
            f_module_path = f_rel_path.replace(sep, ".")
            f_module = pydoc.safeimport(inspect.getmodulename(f_module_path))

            for name, api_module in inspect.getmembers(f_module):
                if inspect.getmodule(api_module) != f_module:
                    continue
                if name.startswith('_'):
                    continue

                # load -> cifar10.load
                substitue_name = f"{f.split('.')[-2]}.{name}"

                self.module_infos.append(
                    [name, api_module, substitue_name, f_rel_path]
                )

    def _solve_name_conflict(self):
        name_count = {}
        for name, _, _, _ in self.module_infos:
            if name in name_count:
                name_count[name] += 1
            else:
                name_count[name] = 1

        duplicate_names = [name for name, count in name_count.items() if count > 1]

        for info in self.module_infos:
            if info[0] in duplicate_names:
                info[0] = info[2]  # use substitute name

    def _build_api_markdown(self):
        for name, mod, _, f_rel_path in self.module_infos:
            APIMarkdown(name=name, module=mod, save_dir=self.save_dir).dump()
            add_source_lines(module=mod,
                             mod_name=name,
                             save_dir=self.save_dir,
                             f_rel_path=f_rel_path)

    def dump(self):
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self._extract_api_module()
        self._solve_name_conflict()
        self._build_api_markdown()


class APIMarkdown:
    def __init__(self, name, module, save_dir):
        self.name = name
        self.module = module
        self.save_dir = save_dir
        self.md_content = None

    def __repr__(self):
        return f"<APIMarkdown name={self.name}, module={self.module}, save_dir={self.save_dir}>"

    def func_to_md(self):
        content = []
        content.append('\n\n')
        content.append('### ' + self.name)
        content.append("\n```python\n")
        content.append(self.name)
        signature = str(inspect.signature(self.module))
        content.append(self.prettify_signature(signature))
        content.append('\n')
        content.append('```')
        content.append('\n')
        content.append(self.format_docstring(inspect.getdoc(self.module)))

        self.md_content = "".join(content)


    def class_to_md(self):
        content = []
        content.append('## ' + self.name)
        content.append("\n```python\n")
        content.append(self.name)
        signature = str(inspect.signature(self.module))
        content.append(self.prettify_signature(signature))
        content.append('\n')
        content.append('```')
        content.append('\n')
        content.append(self.format_docstring(self.module.__doc__))
        content.extend(self.get_class_method())

        self.md_content = "".join(content)


    def get_class_method(self):
        """It extracts the functions which are functions of the current class that is being

        Returns:
            [list]: It returns the markdown string with object signature appended in the list
        """
        output = list()
        funcs = inspect.getmembers(self.module, inspect.isfunction)
        for f in funcs:
            if f[1].__qualname__.split('.')[0] == self.module.__qualname__:
                if not f[0].startswith("_") and not isDoc(f[1]):
                    output.append('\n\n')
                    output.append('### ' + f[0])
                    output.append("\n```python\n")
                    output.append(f[0])
                    signature = str(inspect.signature(f[1]))
                    output.append(APIMarkdown.prettify_signature(signature))
                    output.append('\n')
                    output.append('```')
                    output.append('\n')
                    output.append(APIMarkdown.format_docstring(f[1].__doc__))

        return output


    def dump(self):
        try:  # some class raise error when running inspect.signature(self.module) such as fe.EarlyStop
            if inspect.isclass(self.module):
                self.class_to_md()
            else:
                self.func_to_md()

            save_path = os.path.join(self.save_dir, self.name + ".md")
            with open(save_path, "w") as f:
                f.write(self.md_content)

        except ValueError:
            pass

    @staticmethod
    def prettify_signature(inp: str):
        out = inp
        out = re.sub(r'^\((.+)\)', r'(\n\t\1\n)', out)
        out = re.sub(r' ->', r'\n->', out)
        out = re.sub(r'(, )([\w\d]+:)', r',\n\t\2', out)
        out = re.sub(r':', r': ', out)

        return out

    @staticmethod
    def format_docstring(docstr):
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
                                title = '#### ' + elements[0].strip()
                                res.append('\n\n' + title + ':\n')
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
    fe_path = os.path.abspath((os.path.join(repo_dir, 'fastestimator')))

    if os.path.exists(save_dir):
        shutil.rmtree(save_dir)

    #insert project path to system path to later detect the modules in project
    sys.path.insert(0, fe_path)
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
                               fe_dir=fe_path)
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
            json_dict["sourceurl"] = sourcelines_dict[os.path.abspath(path)]

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
