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

// Package main implements a server for Greeter service.
package main

import (
	"flag"
	"log"
	"net"
	"os"
	"os/signal"
	"suvvm.work/toad_ocr_engine/rpc"
	"syscall"
	"time"

	"google.golang.org/grpc"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
)

var (
	serv = flag.String("service", "hello_service", "service name")
	host = flag.String("host", "localhost", "listening host")
	port = flag.String("port", "18886", "listening port")
	reg  = flag.String("reg", "http://localhost:2379", "register etcd address")
)

func main() {
	lis, err := net.Listen("tcp", ":" + *port)
	if err != nil {
		log.Fatalf("failed to listen: %v", err)
	}
	err = rpc.Register(*reg, *serv, *host, *port, time.Second*10, 15)
	if err != nil {
		panic(err)
	}
	ch := make(chan os.Signal, 1)
	signal.Notify(ch, syscall.SIGTERM, syscall.SIGINT, syscall.SIGHUP, syscall.SIGQUIT)
	go func() {
		s := <-ch
		log.Printf("receive signal '%v'", s)
		rpc.UnRegister()
		os.Exit(1)
	}()
	s := grpc.NewServer()
	pb.RegisterToadOcrServer(s, &rpc.Server{})
	if err := s.Serve(lis); err != nil {
		log.Fatalf("failed to serve: %v", err)
	}
}