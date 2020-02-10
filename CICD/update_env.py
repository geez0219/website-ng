import boto3
import shutil
from datetime import datetime
import time
import zipfile
import subprocess
import os

#elasticbeanstalk parameters
EB_APP_NAME = 'FE'
EB_ENV_ID = 'e-k4uvwzr86r'
EB_APP_VERSION_LABEL = 'fe-stage-v2' #currently set to stage env

#s3 parameters
S3_APP_DIR = 'fe_web'
S3_BUCKET = 'fastestimator-public-data'

APP_PACKAGE = 'fe_web_v1'
WEB_DIR = 'website-ng'


def zip_package():
    dist = os.path.join(WEB_DIR, 'dist')
    args = ['npm','run','--prefix','../website-ng','build:ssr']
    app_path = subprocess.call(args)
    shutil.make_archive(APP_PACKAGE, 'zip', dist)


def deploy_app():
    try:
        new_app_version = os.path.join(S3_APP_DIR, EB_APP_VERSION_LABEL)
        output_filename = APP_PACKAGE + '.zip'

        #compress the code dir and upload to the s3 bucket
        s3 = boto3.resource('s3')
        s3.Bucket(bucket).upload_file(output_filename, new_app_version)

        eb = boto3.client('elasticbeanstalk')
        response = eb.create_application_version(
            ApplicationName=EB_APP_NAME,
            VersionLabel=EB_APP_VERSION_LABEL,
            SourceBundle = {
            'S3Bucket':S3_BUCKET,
            'S3Key':new_app_version
        },
        AutoCreateApplication=False,
        Process=True
        )
        #add some delay before updating environment
        time.sleep(5)

        #update the deployment environment with new version
        response = eb.update_environment(
            ApplicationName=EB_APP_NAME,
            EnvironmentId=EB_ENV_ID,
            VersionLabel=EB_APP_VERSION_LABEL
        )

        print(response)
    except:
        raise


def desc_app():
    eb = boto3.client('elasticbeanstalk', region_name='us-west-2')
    response = eb.describe_environments(
        ApplicationName='FE',
        VersionLabel='fe-stage-v1',
        EnvironmentIds=['e-k4uvwzr86r',],
        EnvironmentNames=[
        'Fe-stage',
    ],
    )
    print(response)


if __name__ == '__main__':
    #zip_package()
    deploy_app()
    #desc_app()
