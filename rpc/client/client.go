/*
 *
 * Copyright 2015 gRPC authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

// Package main implements a client for Greeter service.
package main

import (
	"context"
	"flag"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/resolver"
	"gorgonia.org/tensor"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/method"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/rpc"
	"suvvm.work/toad_ocr_engine/utils"
	"time"

	"google.golang.org/grpc"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
)

const (
	address     = "localhost:50051"
	defaultName = "world"
)

var (
	svc = flag.String("service", "hello_service", "service name")
	reg = flag.String("reg", "http://localhost:2379", "register etcd address")
)

func main() {
	flag.Parse()
	r := rpc.NewResolver(*reg, *svc)
	resolver.Register(r)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	conn, err := grpc.DialContext(ctx, r.Scheme()+"://authority/"+*svc, grpc.WithInsecure(), grpc.WithBalancerName(roundrobin.Name), grpc.WithBlock())
	defer cancel()
	if err != nil {
		panic(err)
	}
	defer conn.Close()

	reader, err := os.Open(common.MNISTTestImagesPath)
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testImages, err := utils.ReadImageFile(reader)
	if err != nil {
		log.Fatal(err)
	}
	rows := len(testImages)	// 矩阵宽度
	cols := len(testImages[0])	// 矩阵高度
	// 创建矩阵的支撑平面切片，tensor会复用当前切片的空间
	supportSlice := make([]float64, 0, rows * cols)
	for i := 0; i < rows; i++ {	// 复制缩放后的像素切片进入矩阵平面切片
		for j := 0; j < len(testImages[i]); j++ {
			supportSlice = append(supportSlice, utils.PixelWeight(testImages[i][j]))
		}
	}
	var images tensor.Tensor
	images = tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(supportSlice))
	var oneimg tensor.Tensor

	if oneimg, err = images.Slice(model.MakeRS(5, 6)); err != nil {
		log.Fatalf("Unable to slice image")
	}
	log.Printf("row %v", rows)
	log.Printf("col %v", cols)
	log.Printf("dims %v", oneimg.Dims())
	log.Printf("shap %v", oneimg.Shape())

	err = method.Visualize(oneimg, 1, 1, "rpc_image.png")
	if err != nil {
		log.Printf("client visualize err:%v", err)
	}
	ticker := time.NewTicker(5000 * time.Millisecond)
	for t := range ticker.C {
		client := pb.NewToadOcrClient(*conn)
		resp, err := client.Predict(context.Background(), &pb.PredictRequest{Image: oneimg.Data().([]float64)})
		if err == nil {
			log.Printf("%v: Msg is %s\nCode is %s\nLab is %s", t, resp.Msg, resp.Code, resp.Label)
		}
	}
}