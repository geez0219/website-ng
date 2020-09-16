import os
import sys

if __name__ == "__main__":
    repo_path = sys.argv[1]
    output_path = sys.argv[2]

    scripts = ["fe_parser.py", "apphub_parser.py", "tutorial_parser.py", "install_parser.py"]
    for script in scripts:
        script_path = os.path.abspath(os.path.join(__file__, "..", script))
        os.system("python {} {} {}".format(script_path, repo_path, output_path))

    resource_path = os.path.join(repo_path, "tutorial", "resources")
    os.system("cp -r {} {}".format(resource_path, output_path))
