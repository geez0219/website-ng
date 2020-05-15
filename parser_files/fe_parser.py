""" FastEstimator Docstring parser """
import inspect
import json
import os
import pydoc
import re
import sys
import tempfile
from os import sep

from pygit2 import Repository

titles = ['Args', 'Raises', 'Returns']
fe_url = 'https://github.com/fastestimator/fastestimator/blob'
sourcelines_dict = {}
html_char_regex = r'([\<\>])'
args_regex = r'(Args|Returns|Raises):\n'


def extractmarkdown(module, save_path, mod_dir, branch_name):
    output = list()
    mod = pydoc.safeimport(inspect.getmodulename(module))
    output.append('## ' + str(module))
    output.append("***\n")
    getclasses(mod, save_path, mod_dir, branch_name)
    getfunctions(mod, save_path, mod_dir, branch_name)
    return "".join(output)


def formatDocstring(docstr):
    """It format the docstring in markdown form and append it to the list

    Args:
        docstr[str]: Input docstring that needs to be formated

    Returns:
        [str]: markdown string converted from docstring
    """
    docstr_output = ''
    if docstr != None:
        docstr_output = inspect.cleandoc(docstr)
        docstr_output = docstr_output.replace('<', '&lt;').replace('>', '&gt;')  # html espace characters to be replaced
        args = re.search(args_regex, docstr_output)
        if args != None:
            pos = args.start(0)
            doc_arg = docstr_output[pos:]
            docstr_output = docstr_output[:pos - 1]
            print(docstr_output)
            new_docstr = doc_arg.split('\n')
            res = []
            if len(new_docstr) != 0:
                for idx in range(len(new_docstr)):
                    if ':' in new_docstr[idx]:
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


def addSourceLines(mod_def, mod_name, mod_dir, branch_name):
    sourcelines = inspect.getsourcelines(mod_def)
    start = sourcelines[1]
    end = start + len(sourcelines[0]) - 1
    sourcelines_dict[mod_name] = os.path.join(
        *[fe_url, branch_name, 'fastestimator', mod_dir + '#L' + str(start) + '-L' + str(end)])


def getclasses(item, save_path, mod_dir, branch_name):
    classes = inspect.getmembers(item, inspect.isclass)
    for cl in classes:
        if inspect.getmodule(cl[1]) == item and not cl[0].startswith("_"):
            try:
                output = list()
                output.append('## ' + cl[0])
                output.append("\n```python\n")
                output.append(cl[0])
                output.append(str(inspect.signature(cl[1])))
                output.append('\n')
                output.append('```')
                output.append('\n')
                output.append(formatDocstring(cl[1].__doc__))
                output.extend(getClassFunctions(cl[1], mod_dir, branch_name))
                addSourceLines(cl[1], cl[0], mod_dir, branch_name)
                with open(os.path.join(save_path, cl[0]) + '.md', 'w') as f:
                    f.write("".join(output))
            except ValueError:
                continue
            getclasses(cl[1], save_path, mod_dir, branch_name)


def getfunctions(item, save_path, mod_dir, branch_name):
    """This function extract the docstring for the function and append the signature and formated markdown to the list before
    storing it to the file

    Args:
        item: Object such as python class or module from which function members are retrieved
        save_path[str]: Path where the file needs to be stored

    """
    funcs = inspect.getmembers(item, inspect.isfunction)
    for f in funcs:
        if inspect.getmodule(f[1]) == inspect.getmodule(item):
            if not f[0].startswith("_"):
                output = list()
                output.append('\n\n')
                output.append('### ' + f[0])
                output.append("\n```python\n")
                output.append(f[0])
                output.append(str(inspect.signature(f[1])))
                output.append('\n')
                output.append('```')
                output.append('\n')
                output.append(formatDocstring(inspect.getdoc(f[1])))
                addSourceLines(f[1], f[0], mod_dir, branch_name)
                with open(os.path.join(save_path, f[0]) + '.md', 'w') as f:
                    f.write("".join(output))


def isDoc(obj):
    doc = obj.__doc__
    if doc == '' or doc == None:
        return True
    return False


def getClassFunctions(item, mod_dir, branch_name):
    """It extracts the functions which are functions of the current class that is being

    Returns:
        [list]: It returns the markdown string with object signature appended in the list
    """
    output = list()
    funcs = inspect.getmembers(item, inspect.isfunction)
    for f in funcs:
        if inspect.getmodule(f[1]) == inspect.getmodule(item):
            if not f[0].startswith("_") and not isDoc(f[1]):
                output.append('\n\n')
                output.append('### ' + f[0])
                output.append("\n```python\n")
                output.append(f[0])
                output.append(str(inspect.signature(f[1])))
                output.append('\n')
                output.append('```')
                output.append('\n')
                output.append(formatDocstring(f[1].__doc__))
                addSourceLines(f[1], f[0], mod_dir, branch_name)

    return output


def generatedocs(repo_dir, save_dir):
    """This function loop through files and sub-directories in project directory in top down approach and get python code
    files to extract markdowns. It also prepares path to save markdown file for corresponding python file.

    Args:
        path[str]: Path to the project which markdown string are extracted

    Returns:
        [str]: Returns absolute path to the generated markdown directory
    """
    main_repo = os.path.join(repo_dir, 'fastestimator')
    head = Repository(repo_dir).head
    #branch_name = head.name.split(sep)[-1]
    branch_name = 'r1.0'
    fe_path = os.path.abspath(main_repo)
    save_dir = os.path.join(save_dir, 'fe')
    #insert project path to system path to later detect the modules in project
    sys.path.insert(0, fe_path)
    #parent directory where all the markdown files will be stored

    for subdirs, dirs, files in os.walk(fe_path, topdown=True):
        for f in files:
            fname, ext = os.path.splitext(os.path.basename(f))
            if not f.startswith('_') and ext == '.py':
                #if f == 'cross_entropy.py':
                f_path = os.path.join(subdirs, f)
                mod_dir = os.path.relpath(f_path, fe_path)
                mod = mod_dir.replace(sep, '.')
                if subdirs == fe_path:
                    save_path = os.path.join(*[save_dir, 'fe'])
                else:
                    save_path = os.path.join(*[save_dir, os.path.relpath(subdirs, fe_path)])
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                mdtexts = extractmarkdown(mod, save_path, mod_dir, branch_name)
    return save_dir


def generate_json(path):
    """This function generates JSON file that represents the file structure of the markdown files. JSON file is rendered along
    with the markdown files to create API webpages.

    Args:
        path[str]: Path to the generated markdown files

    Returns:
        [list]: list which contains file structure of the markdown files
    """

    doc_path = path  #keep a copy of the path to later use it in recursive calls

    def createlist(path):
        name = os.path.relpath(path, doc_path)
        displayname = os.path.basename(name).split('.')[0]
        json_dict = {'name': name.strip('.,')}
        prefix = 'fe.'
        if os.path.isdir(path):
            name = name.replace(sep, '.')
            json_dict['displayName'] = 'fe'
            if displayname != '' and name != 'fe':
                json_dict['displayName'] = prefix + name
            children = [createlist(os.path.join(path, x)) for x in sorted(os.listdir(path))]

            subfield, field = [], []
            for x in children:
                if 'children' in x:
                    subfield.append(x)
                else:
                    field.append(x)
            field = sorted(field, key=lambda x: x['displayName'])
            subfield.extend(field)
            json_dict['children'] = subfield
        else:
            json_dict['displayName'] = displayname
            json_dict['sourceurl'] = sourcelines_dict[displayname]
        return json_dict

    if os.path.isdir(path):
        json_list = [createlist(os.path.join(path, x)) for x in os.listdir(path)]
        json_list = sorted(json_list, key=lambda x: x['displayName'])
        return json_list


if __name__ == '__main__':

    tmp_output = sys.argv[2]
    save_dir = os.path.join(tmp_output, 'api')
    docs_path = generatedocs(sys.argv[1], save_dir)
    #print(sourcelines_dict)
    struct_json = os.path.join(save_dir, 'structure.json')
    with open(struct_json, 'w') as f:
        fe_json = json.dumps(generate_json(docs_path))
        f.write(fe_json)
