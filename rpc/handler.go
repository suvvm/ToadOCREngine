package rpc

import (
	"context"
	"flag"
	"gorgonia.org/gorgonia"
	"log"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/nn"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
)

var (
	successMsg = flag.String("success msg", "success", "rpc reply message")
	successCode = flag.Int("success code", 0, "rpc reply code")
	errorCode = flag.Int("error code", 1, "rpc reply code")
	errorLab = flag.String("error label", "-1", "rpc reply label")
	cnn *model.CNN
	snn *model.SNN
)

func initNN() {
	var err error
	snn, err = model.LoadSNNFromSave()
	if err != nil {
		log.Fatalf("Failed at load snn weights %v", err)
	}
	cnn, err = model.LoadCNNFromSave()
	defer cnn.VM.Close()
	if err != nil {
		log.Fatalf("Unable to load cnn file %v", err)
	}
	var out gorgonia.Value
	rv := gorgonia.Read(cnn.Out, &out)
	fwdNet := cnn.G.SubgraphRoots(rv)
	vm := gorgonia.NewTapeMachine(fwdNet)
	cnn.VM = vm
}

// server is used to implement helloworld.GreeterServer.
type Server struct {
	pb.UnimplementedToadOcrServer
}

// SayHello implements helloworld.GreeterServer
func (s *Server) SayHello(ctx context.Context, in *pb.HelloRequest) (*pb.HelloReply, error) {
	log.Printf("Received: %v", in.GetName())
	return &pb.HelloReply{Message: "Hello " + in.GetName()}, nil
}

func (s *Server) Predict(ctx context.Context, in *pb.PredictRequest) (*pb.PredictReply, error) {
	log.Printf("Predict %v", in.NetFlag)
	var lab string
	var err error
	if in.NetFlag == common.SnnName {
		lab, err = nn.SnnPredict(snn, in.Image)
	} else if in.NetFlag == common.CnnName {
		lab, err = nn.CnnPredict(cnn, in.Image)
	}
	if err != nil {
		return &pb.PredictReply{Code: int32(*errorCode), Message: err.Error(), Label: *errorLab}, nil
	}
	return &pb.PredictReply{Code: int32(*successCode), Message: *successMsg, Label: lab}, nil
}
