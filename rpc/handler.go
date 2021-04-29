package rpc

import (
	"context"
	"log"
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
