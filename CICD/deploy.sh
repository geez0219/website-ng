#!/bin/bash
TMP_DIR=../../../tmp_output
ANGULAR_ASSETS=$WORKSPACE/website-ng/src/assets
echo $TMP_DIR
echo $ANGULAR_ASSETS
rm -rf ../../fastestimator
mkdir ../../fastestimator
if [[ ! -e $TMP_DIR ]]; then
        mkdir $TMP_DIR
fi
git clone https://github.com/Vivek305/fastestimator.git ../../fastestimator/
#source ../../venv/bin/activate
python3 ../../fastestimator/docs/fe_parser.py
python3 ../../fastestimator/docs/tutorial_parser.py
python3 ../../fastestimator/docs/apphub_parser.py
echo 'finished parsing'
rm -rf $ANGULAR_ASSETS/api && cp -rf $TMP_DIR/api $ANGULAR_ASSETS/
rm -rf $ANGULAR_ASSETS/tutorial && cp -rf $TMP_DIR/tutorials $ANGULAR_ASSETS/
rm -rf $ANGULAR_ASSETS/example && cp -rf $TMP_DIR/example $ANGULAR_ASSETS/