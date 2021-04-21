#!/bin/bash

RUN_NAME="toad_ocr_engine"

# 判断GO环境
if [ -n "$GOPATH" ];then
  rm -rf ./output/*  # 清空out目录
  rm -f ${RUN_NAME}
  mkdir -p ./output/bin # 创建二进制文件存放目录
  mkdir -p ./output/images # 创建图像文件存放目录
  cp -r ./resources ./output/resources
  go build -o ./output/bin/${RUN_NAME}
  cp ./output/resources/script/bootstrap.sh ./output/bootstrap.sh
  cp ./output/bin/${RUN_NAME} ${RUN_NAME}
  chmod +x ./output/bootstrap.sh
else
	echo "GOPATH is needed!"
fi
