package rpc

import (
	"flag"
	"google.golang.org/grpc"
	"log"
	"net"
	"os"
	"os/signal"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
	"syscall"
	"time"
)

var (
	serv = flag.String("service", "toad_ocr_service", "service name")
	host = flag.String("host", "localhost", "listening host")
	port = flag.String("port", "18886", "listening port")
	reg  = flag.String("reg", "http://localhost:2379", "register etcd address")
)

func RunRPCServer() {
	initNN()
	log.Printf("service listen port:%v", port)
	lis, err := net.Listen("tcp", ":" + *port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	log.Printf("register rpc server to control center...")
	err = Register(*reg, *serv, *host, *port, time.Second*10, 15)
	if err != nil {
		panic(err)
	}
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP, syscall.SIGQUIT)
	go func() {
		s := <-ch
		log.Printf("receive signal '%v'", s)
		UnRegister()
		os.Exit(1)
	}()
	log.Printf("create new toad ocr rpc server...")
	s := grpc.NewServer()
	log.Printf("register handler...")
	pb.RegisterToadOcrServer(s, &Server{})
	log.Printf("run toad ocr rpc server...")
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}

