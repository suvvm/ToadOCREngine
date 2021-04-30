#!/bin/bash
# start a local etcd server
~/etcd/etcd-download-test/etcd

# write,read to etcd
ETCDCTL_API=3 ~/etcd/etcd-download-test/etcdctl --endpoints=localhost:2379 put foo bar
ETCDCTL_API=3 ~/etcd/etcd-download-test/etcdctl --endpoints=localhost:2379 get foo

