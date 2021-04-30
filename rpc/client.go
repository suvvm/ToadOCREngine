package rpc

import (
	"context"
	"flag"
	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/resolver"
	"gorgonia.org/tensor"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/method"
	"suvvm.work/toad_ocr_engine/model"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
	"suvvm.work/toad_ocr_engine/utils"
	"time"
)

func RunRpcClient() {
	flag.Parse()
	r := NewResolver(*reg, *serv)
	resolver.Register(r)
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	conn, err := grpc.DialContext(ctx, r.Scheme()+"://authority/"+*serv, grpc.WithInsecure(), grpc.WithBalancerName(roundrobin.Name), grpc.WithBlock())
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
	client := pb.NewToadOcrClient(*conn)
	resp, err := client.Predict(context.Background(), &pb.PredictRequest{NetFlag: common.SnnName, Image: oneimg.Data().([]float64)})
	if err == nil {
		log.Printf("\nSNN\nMsg is %s\nCode is %d\nLab is %s", resp.Message, resp.Code, resp.Label)
	}
	resp, err = client.Predict(context.Background(), &pb.PredictRequest{NetFlag: common.CnnName, Image: oneimg.Data().([]float64)})
	if err == nil {
		log.Printf("\nCNN\nMsg is %s\nCode is %d\nLab is %s", resp.Message, resp.Code, resp.Label)
	}
}
