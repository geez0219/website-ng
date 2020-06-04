import os
import sys


def parse_install(fe_path):
    install_filepath = os.path.join(fe_path, 'README.md')
    install_content = open(install_filepath).readlines()
    startidx = 0
    endidx = len(install_content) - 1
    for line in install_content:
        if '## Prerequisites' in line:
            startidx = install_content.index(line)
        elif '## Useful Links' in line:
            endidx = install_content.index(line)

    web_install_content = install_content[startidx:endidx - 1]
    return ''.join(web_install_content).rstrip()


if __name__ == "__main__":
    fe_path = sys.argv[1]
    output_path = sys.argv[2]
    output_path = os.path.abspath(output_path)
    fe_path = os.path.abspath(fe_path)
    install_content = parse_install(fe_path)
    # save install markdown
    install_md = os.path.join(output_path, 'install.md')
    with open(install_md, 'w') as f:
        f.write(install_content)
