package cluster

import (
	"context"
	"fmt"
	"github.com/coreos/etcd/clientv3"
	"log"
	"strings"
	"time"
)

var (
	cli  *clientv3.Client
)

func init() {
	var err error
	cli, err = clientv3.New(clientv3.Config{
		Endpoints: strings.Split("http://localhost:2379", ","),
		DialTimeout: 10 * time.Second,
	})
	if err != nil {
		log.Fatalf("init cluster err:%v", err)
	}
}

func GetKV(ctx context.Context, key string) (string, error) {
	clusterCtx, cancel := context.WithTimeout(ctx, 5 * time.Second)
	defer cancel()
	resp, err := cli.Get(clusterCtx, key)
	if err != nil {
		return "", err
	}
	if len(resp.Kvs) == 0{
		return "", fmt.Errorf("resp kvs is empty")
	}
	//for _, ev := range resp.Kvs {
	//	fmt.Printf("%s : %s\n", ev.Key, ev.Value)
	//}
	return string(resp.Kvs[0].Value), nil
}

func PutKV(ctx context.Context, key, value string) error {
	clusterCtx, cancel := context.WithTimeout(ctx, 5 * time.Second)
	defer cancel()
	_, err := cli.Put(clusterCtx, key, value)
	if err != nil {
		return err
	}
	return nil
}

func DelKV(ctx context.Context, key string) error {
	if _, err := GetKV(ctx, key); err != nil {
		return err
	}
	clusterCtx, cancel := context.WithTimeout(ctx, 5 * time.Second)
	defer cancel()
	_, err := cli.Delete(clusterCtx, key)
	if err != nil {
		return err
	}
	return nil
}
