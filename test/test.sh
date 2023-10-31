#!/usr/bin/env bash

WORK_DIR=/data/tests

python3 $WORK_DIR/integration/test_tesseract.py
python3 $WORK_DIR/integration/test_elasticsearch.py

exit 0