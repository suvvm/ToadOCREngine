package rpc

import (
	"context"
	"flag"
	cpb "github.com/cheggaaa/pb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/balancer/roundrobin"
	"google.golang.org/grpc/resolver"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf64"
	"log"
	"os"
	"strconv"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/model"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
	"suvvm.work/toad_ocr_engine/utils"
	"time"
)

func RunRpcClient() {
	var err error
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
	client := pb.NewToadOcrClient(*conn)

	reader, err := os.Open(common.EMNISTByClassTestImagesPath)
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testImages, err := utils.ReadImageFile(reader)
	if err != nil {
		log.Fatal(err)
	}
	reader, err = os.Open(common.EMNISTByClassTestLabelsPath)
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testlabels, err := utils.ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	dataImgs := utils.PrepareX(testImages)
	datalabs := utils.PrepareY(testlabels, common.EMNISTByClassNumLabels)


	// shape := dataImgs.Shape()
	var correct, total float64
	var image, label tensor.Tensor
	var errCnt int
	// bar := pb.New(tmpDataSize)
	bar := cpb.New(200)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(common.BarMaxWidth)
	bar.Prefix("Testing")
	bar.Set(0)
	bar.Start()
	// for i := 0; i < tmpDataSize; i++ {
	for i := 0; i < 200; i++ {
		if image, err = dataImgs.Slice(model.MakeRS(i, i + 1)); err != nil {
			log.Fatalf("Unable to slice image %d", i)
		}
		if label, err = datalabs.Slice(model.MakeRS(i, i+1)); err != nil {
			log.Fatalf("Unable to slice label %d", i)
		}
		label := vecf64.Argmax(label.Data().([]float64))
		resp, err := client.Predict(context.Background(), &pb.PredictRequest{NetFlag: common.SnnName, Image: image.Data().([]float64)})
		if err != nil {
			log.Printf("error:%v", err)
		}
		respInt, _ := strconv.Atoi(resp.Label)
		if respInt == label {
			correct++
		} else {
			errCnt++
		}
		total++
		bar.Increment()
	}
	bar.Finish()
	log.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
}
