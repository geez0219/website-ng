import json
import os
import shutil
import subprocess
import time
import zipfile
import sys
from datetime import datetime

import boto3

#elasticbeanstalk parameters
EB_APP_NAME = 'FE'
EB_ENV_ID = 'e-k4uvwzr86r'
EB_APP_VERSION_LABEL = 'fe-stage-v2'  #currently set to stage env

#s3 parameters
S3_APP_DIR = 'fe_web'
S3_BUCKET = 'fastestimator-public-data'

APP_PACKAGE = 'fe_web_v1'
WEB_DIR = 'website-ng'


def getDeployConfig(deploy_config_path):
    with open(deploy_config_path, 'r') as f:
        config = json.load(f)
        return config


def saveDeployConfig(config, deploy_config_path):
    with open(deploy_config_path, 'w') as f:
        json.dump(config, f)


def zip_package(web_dir):
    dist = os.path.join(web_dir, 'dist')
    args = ['npm', 'run', '--prefix', web_dir, 'build:ssr']
    app_path = subprocess.call(args)
    shutil.make_archive(APP_PACKAGE, 'zip', dist)


def deploy_app(config, deploy_config_path):
    try:
        app_version_label = config['app_version_label']
        app_name = config['app_name']
        app_env_id = config['app_env_id']

        app_version_segments = app_version_label.split('.')
        cur_version = int(app_version_segments[-1]) + 1
        app_version_segments[-1] = str(cur_version)
        app_version_label = '.'.join(app_version_segments)

        new_app_version = os.path.join(S3_APP_DIR, app_version_label)
        output_filename = APP_PACKAGE + '.zip'

        #compress the code dir and upload to the s3 bucket
        s3 = boto3.resource('s3')
        s3.Bucket(S3_BUCKET).upload_file(output_filename, new_app_version)

        eb = boto3.client('elasticbeanstalk')
        response = eb.create_application_version(
            ApplicationName=app_name,
            VersionLabel=app_version_label,
            SourceBundle={
                'S3Bucket': S3_BUCKET,
                'S3Key': new_app_version
            },
            AutoCreateApplication=False,
            Process=True)
        #add some delay before updating environment
        time.sleep(5)

        #update the deployment environment with new version
        response = eb.update_environment(ApplicationName=app_name,
                                         EnvironmentId=app_env_id,
                                         VersionLabel=app_version_label)
        config['app_version_label'] = app_version_label
        config['app_name'] = app_name
        config['app_env_id'] = app_env_id
        saveConfig(config, deploy_config_path)
        print(response)
    except:
        raise


def desc_app():
    eb = boto3.client('elasticbeanstalk', region_name='us-west-2')
    response = eb.describe_environments(
        ApplicationName='FE',
        VersionLabel='fe-stage-v1',
        EnvironmentIds=[
            'e-k4uvwzr86r',
        ],
        EnvironmentNames=[
            'Fe-stage',
        ],
    )
    print(response)


if __name__ == '__main__':
    web_dir = sys.argv[1]
    deploy_config_path = sys.argv[2]
    config = getDeployConfig(deploy_config_path)
    zip_package(web_dir)
    deploy_app(config, deploy_config_path)
    #desc_app()
