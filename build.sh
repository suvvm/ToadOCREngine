#!/bin/bash

RUN_NAME="toad_ocr_engine"

# 判断GO环境
if [ -n "$GOPATH" ];then
  rm -rf ./output/*  # 清空out目录
  rm -f ${RUN_NAME}
  mkdir -p ./output/bin # 创建二进制文件存放目录
  mkdir -p ./output/images # 创建图像文件存放目录
  cp -r ./resources ./output/resources
  go build -tags='CUDA' -o ./output/bin/${RUN_NAME}
  cp ./output/resources/script/bootstrap.sh ./output/bootstrap.sh
  cp ./output/bin/${RUN_NAME} ${RUN_NAME}
  chmod +x ./output/bootstrap.sh
  cd resources
  if [ ! -d "mnist/" ]; then
    curl -o mnist.zip 'https://www.suvvm.work/sundry/mnist.zip' \
    -H 'Connection: keep-alive' \
    -H 'Pragma: no-cache' \
    -H 'Cache-Control: no-cache' \
    -H 'Upgrade-Insecure-Requests: 1' \
    -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36' \
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
    -H 'Sec-Fetch-Site: none' \
    -H 'Sec-Fetch-Mode: navigate' \
    -H 'Sec-Fetch-User: ?1' \
    -H 'Sec-Fetch-Dest: document' \
    -H 'sec-ch-ua: "Google Chrome";v="89", "Chromium";v="89", ";Not A Brand";v="99"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'Accept-Language: zh-CN,zh;q=0.9' \
    -H 'Cookie: Hm_lvt_e5190116d64be66cac29daef18c94ac0=1616226568; serverType=nginx; order=id%20desc; uploadSize=1073741824; Hm_lpvt_e5190116d64be66cac29daef18c94ac0=1618059473; rank=a; BT_PANEL=f0c0ee9844553113db39e73ece79446fed35d7fa; memRealUsed=714; mem-before=714/1839%20%28MB%29; memSize=1839; five=0.13; fifteen=0.1; conterError=; upNet=2.42; downNet=4.27; one=0.05; Path=/www/wwwroot/www.suvvm.work/suvvm.github.io/sundry' \
    --compressed
    unzip mnist.zip
    rm -f mnist.zip
  fi
  if [ ! -d "emnist/" ]; then
    curl -o emnist-byclass.zip 'https://www.suvvm.work/sundry/emnist-byclass.zip' \
    -H 'Connection: keep-alive' \
    -H 'Pragma: no-cache' \
    -H 'Cache-Control: no-cache' \
    -H 'Upgrade-Insecure-Requests: 1' \
    -H 'User-Agent: Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36' \
    -H 'Accept: text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9' \
    -H 'Sec-Fetch-Site: none' \
    -H 'Sec-Fetch-Mode: navigate' \
    -H 'Sec-Fetch-User: ?1' \
    -H 'Sec-Fetch-Dest: document' \
    -H 'sec-ch-ua: "Google Chrome";v="89", "Chromium";v="89", ";Not A Brand";v="99"' \
    -H 'sec-ch-ua-mobile: ?0' \
    -H 'Accept-Language: zh-CN,zh;q=0.9' \
    -H 'Cookie: serverType=nginx; order=id%20desc; uploadSize=1073741824; rank=a; BT_PANEL=f0c0ee9844553113db39e73ece79446fed35d7fa; memSize=1839; conterError=; Path=/var/log; Hm_lvt_e5190116d64be66cac29daef18c94ac0=; Hm_lpvt_e5190116d64be66cac29daef18c94ac0=1619088262; memRealUsed=708; mem-before=708/1839%20%28MB%29; fifteen=0.1; one=0.01; five=0.06; upNet=0.64; downNet=0.85' \
    --compressed
    unzip emnist-byclass.zip
    rm -f emnist-byclass.zip
  fi
else
	echo "GOPATH is needed!"
fi

#cd ..
#chmod +x idl_generate.sh
#sh idl_generate.sh
