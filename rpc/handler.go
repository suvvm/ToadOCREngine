package rpc

import (
	"context"
	"log"
	"suvvm.work/toad_ocr_engine/nn"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
)

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
	log.Printf("predict %v", in.NetFlag)
	lab, err := nn.SnnPredict(in.Image)
	if err != nil {
		return &pb.PredictReply{Code: "1", Msg: "err", Label: "null"}, nil
	}
	return &pb.PredictReply{Code: "0", Msg: "test", Label: lab}, nil
}
