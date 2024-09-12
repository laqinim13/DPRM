#!/usr/bin/env bash
#pip install cython==3.0.2 -i https://mirrors.aliyun.com/pypi/simple/
cd pose/utils
rm -rf nms/*.so
make;cd ../../
