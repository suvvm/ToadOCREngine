#!/bin/bash

which protoc
if [ "$?" -ne 0 ]; then
  echo"protobuf is needed! please install protobuf first!"
else
	protoc -I rpc/idl rpc/idl/toad_ocr.proto --go_out=plugins=grpc:rpc/idl
  sed -i '' 's/ClientConnInterface/ClientConn/g' rpc/idl/toad_ocr.pb.go
  sed -i '' 's/SupportPackageIsVersion6/SupportPackageIsVersion4/g' rpc/idl/toad_ocr.pb.go
fi
