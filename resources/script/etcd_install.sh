#!/bin/bash
ETCD_VER=v3.2.32

# choose either URL
GOOGLE_URL=https://storage.googleapis.com/etcd
GITHUB_URL=https://github.com/etcd-io/etcd/releases/download
DOWNLOAD_URL=${GITHUB_URL}

rm -f ~/etcd/etcd-${ETCD_VER}-darwin-amd64.zip
rm -rf ~/etcd/etcd-download-test && mkdir -p ~/etcd/etcd-download-test

curl -L ${DOWNLOAD_URL}/${ETCD_VER}/etcd-${ETCD_VER}-darwin-amd64.zip -o ~/etcd/etcd-${ETCD_VER}-darwin-amd64.zip
unzip ~/etcd/etcd-${ETCD_VER}-darwin-amd64.zip -d ~/etcd && rm -f ~/etcd/etcd-${ETCD_VER}-darwin-amd64.zip
mv ~/etcd/etcd-${ETCD_VER}-darwin-amd64/* ~/etcd/etcd-download-test && rm -rf mv ~/etcd/etcd-${ETCD_VER}-darwin-amd64

~/etcd/etcd-download-test/etcd --version
ETCDCTL_API=3 ~/etcd/etcd-download-test/etcdctl version

