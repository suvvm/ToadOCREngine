#!/bin/bash

chmod +x ./resources/script/etcd_install.sh
chmod +x ./resources/script/etcd_start.sh
if [ ! -d "~/etcd" ]; then
  mkdir ~/etcd
  sh ./resources/script/etcd_install.sh
fi
sh ./resources/script/etcd_start.sh
