// Copyright 2015 gRPC authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.26.0
// 	protoc        v3.15.8
// source: toad_ocr.proto

package __

import (
	context "context"
	grpc "google.golang.org/grpc"
	codes "google.golang.org/grpc/codes"
	status "google.golang.org/grpc/status"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

// The request message containing the user's name.
type HelloRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	AppId      string `protobuf:"bytes,1,opt,name=app_id,json=appId,proto3" json:"app_id,omitempty"`
	BasicToken string `protobuf:"bytes,2,opt,name=basic_token,json=basicToken,proto3" json:"basic_token,omitempty"`
	Name       string `protobuf:"bytes,3,opt,name=name,proto3" json:"name,omitempty"`
}

func (x *HelloRequest) Reset() {
	*x = HelloRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_toad_ocr_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *HelloRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HelloRequest) ProtoMessage() {}

func (x *HelloRequest) ProtoReflect() protoreflect.Message {
	mi := &file_toad_ocr_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HelloRequest.ProtoReflect.Descriptor instead.
func (*HelloRequest) Descriptor() ([]byte, []int) {
	return file_toad_ocr_proto_rawDescGZIP(), []int{0}
}

func (x *HelloRequest) GetAppId() string {
	if x != nil {
		return x.AppId
	}
	return ""
}

func (x *HelloRequest) GetBasicToken() string {
	if x != nil {
		return x.BasicToken
	}
	return ""
}

func (x *HelloRequest) GetName() string {
	if x != nil {
		return x.Name
	}
	return ""
}

// The response message containing the greetings
type HelloReply struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Message string `protobuf:"bytes,1,opt,name=message,proto3" json:"message,omitempty"`
}

func (x *HelloReply) Reset() {
	*x = HelloReply{}
	if protoimpl.UnsafeEnabled {
		mi := &file_toad_ocr_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *HelloReply) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*HelloReply) ProtoMessage() {}

func (x *HelloReply) ProtoReflect() protoreflect.Message {
	mi := &file_toad_ocr_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use HelloReply.ProtoReflect.Descriptor instead.
func (*HelloReply) Descriptor() ([]byte, []int) {
	return file_toad_ocr_proto_rawDescGZIP(), []int{1}
}

func (x *HelloReply) GetMessage() string {
	if x != nil {
		return x.Message
	}
	return ""
}

type PredictRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	AppId      string `protobuf:"bytes,1,opt,name=app_id,json=appId,proto3" json:"app_id,omitempty"`
	BasicToken string `protobuf:"bytes,2,opt,name=basic_token,json=basicToken,proto3" json:"basic_token,omitempty"`
	NetFlag    string `protobuf:"bytes,3,opt,name=net_flag,json=netFlag,proto3" json:"net_flag,omitempty"`
	Image      []byte `protobuf:"bytes,4,opt,name=image,proto3" json:"image,omitempty"`
}

func (x *PredictRequest) Reset() {
	*x = PredictRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_toad_ocr_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PredictRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PredictRequest) ProtoMessage() {}

func (x *PredictRequest) ProtoReflect() protoreflect.Message {
	mi := &file_toad_ocr_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PredictRequest.ProtoReflect.Descriptor instead.
func (*PredictRequest) Descriptor() ([]byte, []int) {
	return file_toad_ocr_proto_rawDescGZIP(), []int{2}
}

func (x *PredictRequest) GetAppId() string {
	if x != nil {
		return x.AppId
	}
	return ""
}

func (x *PredictRequest) GetBasicToken() string {
	if x != nil {
		return x.BasicToken
	}
	return ""
}

func (x *PredictRequest) GetNetFlag() string {
	if x != nil {
		return x.NetFlag
	}
	return ""
}

func (x *PredictRequest) GetImage() []byte {
	if x != nil {
		return x.Image
	}
	return nil
}

type PredictReply struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Code    int32  `protobuf:"varint,1,opt,name=code,proto3" json:"code,omitempty"`
	Message string `protobuf:"bytes,2,opt,name=message,proto3" json:"message,omitempty"`
	Label   string `protobuf:"bytes,3,opt,name=label,proto3" json:"label,omitempty"`
}

func (x *PredictReply) Reset() {
	*x = PredictReply{}
	if protoimpl.UnsafeEnabled {
		mi := &file_toad_ocr_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *PredictReply) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*PredictReply) ProtoMessage() {}

func (x *PredictReply) ProtoReflect() protoreflect.Message {
	mi := &file_toad_ocr_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use PredictReply.ProtoReflect.Descriptor instead.
func (*PredictReply) Descriptor() ([]byte, []int) {
	return file_toad_ocr_proto_rawDescGZIP(), []int{3}
}

func (x *PredictReply) GetCode() int32 {
	if x != nil {
		return x.Code
	}
	return 0
}

func (x *PredictReply) GetMessage() string {
	if x != nil {
		return x.Message
	}
	return ""
}

func (x *PredictReply) GetLabel() string {
	if x != nil {
		return x.Label
	}
	return ""
}

var File_toad_ocr_proto protoreflect.FileDescriptor

var file_toad_ocr_proto_rawDesc = []byte{
	0x0a, 0x0e, 0x74, 0x6f, 0x61, 0x64, 0x5f, 0x6f, 0x63, 0x72, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f,
	0x12, 0x03, 0x69, 0x64, 0x6c, 0x22, 0x5a, 0x0a, 0x0c, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x15, 0x0a, 0x06, 0x61, 0x70, 0x70, 0x5f, 0x69, 0x64, 0x18,
	0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x61, 0x70, 0x70, 0x49, 0x64, 0x12, 0x1f, 0x0a, 0x0b,
	0x62, 0x61, 0x73, 0x69, 0x63, 0x5f, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x18, 0x02, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x0a, 0x62, 0x61, 0x73, 0x69, 0x63, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x12, 0x12, 0x0a,
	0x04, 0x6e, 0x61, 0x6d, 0x65, 0x18, 0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x04, 0x6e, 0x61, 0x6d,
	0x65, 0x22, 0x26, 0x0a, 0x0a, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x52, 0x65, 0x70, 0x6c, 0x79, 0x12,
	0x18, 0x0a, 0x07, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09,
	0x52, 0x07, 0x6d, 0x65, 0x73, 0x73, 0x61, 0x67, 0x65, 0x22, 0x79, 0x0a, 0x0e, 0x50, 0x72, 0x65,
	0x64, 0x69, 0x63, 0x74, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x15, 0x0a, 0x06, 0x61,
	0x70, 0x70, 0x5f, 0x69, 0x64, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x61, 0x70, 0x70,
	0x49, 0x64, 0x12, 0x1f, 0x0a, 0x0b, 0x62, 0x61, 0x73, 0x69, 0x63, 0x5f, 0x74, 0x6f, 0x6b, 0x65,
	0x6e, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x0a, 0x62, 0x61, 0x73, 0x69, 0x63, 0x54, 0x6f,
	0x6b, 0x65, 0x6e, 0x12, 0x19, 0x0a, 0x08, 0x6e, 0x65, 0x74, 0x5f, 0x66, 0x6c, 0x61, 0x67, 0x18,
	0x03, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x6e, 0x65, 0x74, 0x46, 0x6c, 0x61, 0x67, 0x12, 0x14,
	0x0a, 0x05, 0x69, 0x6d, 0x61, 0x67, 0x65, 0x18, 0x04, 0x20, 0x01, 0x28, 0x0c, 0x52, 0x05, 0x69,
	0x6d, 0x61, 0x67, 0x65, 0x22, 0x52, 0x0a, 0x0c, 0x50, 0x72, 0x65, 0x64, 0x69, 0x63, 0x74, 0x52,
	0x65, 0x70, 0x6c, 0x79, 0x12, 0x12, 0x0a, 0x04, 0x63, 0x6f, 0x64, 0x65, 0x18, 0x01, 0x20, 0x01,
	0x28, 0x05, 0x52, 0x04, 0x63, 0x6f, 0x64, 0x65, 0x12, 0x18, 0x0a, 0x07, 0x6d, 0x65, 0x73, 0x73,
	0x61, 0x67, 0x65, 0x18, 0x02, 0x20, 0x01, 0x28, 0x09, 0x52, 0x07, 0x6d, 0x65, 0x73, 0x73, 0x61,
	0x67, 0x65, 0x12, 0x14, 0x0a, 0x05, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x18, 0x03, 0x20, 0x01, 0x28,
	0x09, 0x52, 0x05, 0x6c, 0x61, 0x62, 0x65, 0x6c, 0x32, 0x70, 0x0a, 0x07, 0x54, 0x6f, 0x61, 0x64,
	0x4f, 0x63, 0x72, 0x12, 0x30, 0x0a, 0x08, 0x53, 0x61, 0x79, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x12,
	0x11, 0x2e, 0x69, 0x64, 0x6c, 0x2e, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x1a, 0x0f, 0x2e, 0x69, 0x64, 0x6c, 0x2e, 0x48, 0x65, 0x6c, 0x6c, 0x6f, 0x52, 0x65,
	0x70, 0x6c, 0x79, 0x22, 0x00, 0x12, 0x33, 0x0a, 0x07, 0x50, 0x72, 0x65, 0x64, 0x69, 0x63, 0x74,
	0x12, 0x13, 0x2e, 0x69, 0x64, 0x6c, 0x2e, 0x50, 0x72, 0x65, 0x64, 0x69, 0x63, 0x74, 0x52, 0x65,
	0x71, 0x75, 0x65, 0x73, 0x74, 0x1a, 0x11, 0x2e, 0x69, 0x64, 0x6c, 0x2e, 0x50, 0x72, 0x65, 0x64,
	0x69, 0x63, 0x74, 0x52, 0x65, 0x70, 0x6c, 0x79, 0x22, 0x00, 0x42, 0x2f, 0x0a, 0x1a, 0x77, 0x6f,
	0x72, 0x6b, 0x2e, 0x73, 0x75, 0x76, 0x76, 0x6d, 0x2e, 0x74, 0x6f, 0x61, 0x64, 0x5f, 0x6f, 0x63,
	0x72, 0x5f, 0x65, 0x6e, 0x67, 0x69, 0x6e, 0x65, 0x42, 0x0c, 0x54, 0x6f, 0x61, 0x64, 0x4f, 0x63,
	0x72, 0x50, 0x72, 0x6f, 0x74, 0x6f, 0x50, 0x01, 0x5a, 0x01, 0x2f, 0x62, 0x06, 0x70, 0x72, 0x6f,
	0x74, 0x6f, 0x33,
}

var (
	file_toad_ocr_proto_rawDescOnce sync.Once
	file_toad_ocr_proto_rawDescData = file_toad_ocr_proto_rawDesc
)

func file_toad_ocr_proto_rawDescGZIP() []byte {
	file_toad_ocr_proto_rawDescOnce.Do(func() {
		file_toad_ocr_proto_rawDescData = protoimpl.X.CompressGZIP(file_toad_ocr_proto_rawDescData)
	})
	return file_toad_ocr_proto_rawDescData
}

var file_toad_ocr_proto_msgTypes = make([]protoimpl.MessageInfo, 4)
var file_toad_ocr_proto_goTypes = []interface{}{
	(*HelloRequest)(nil),   // 0: idl.HelloRequest
	(*HelloReply)(nil),     // 1: idl.HelloReply
	(*PredictRequest)(nil), // 2: idl.PredictRequest
	(*PredictReply)(nil),   // 3: idl.PredictReply
}
var file_toad_ocr_proto_depIdxs = []int32{
	0, // 0: idl.ToadOcr.SayHello:input_type -> idl.HelloRequest
	2, // 1: idl.ToadOcr.Predict:input_type -> idl.PredictRequest
	1, // 2: idl.ToadOcr.SayHello:output_type -> idl.HelloReply
	3, // 3: idl.ToadOcr.Predict:output_type -> idl.PredictReply
	2, // [2:4] is the sub-list for method output_type
	0, // [0:2] is the sub-list for method input_type
	0, // [0:0] is the sub-list for extension type_name
	0, // [0:0] is the sub-list for extension extendee
	0, // [0:0] is the sub-list for field type_name
}

func init() { file_toad_ocr_proto_init() }
func file_toad_ocr_proto_init() {
	if File_toad_ocr_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_toad_ocr_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*HelloRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_toad_ocr_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*HelloReply); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_toad_ocr_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PredictRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_toad_ocr_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*PredictReply); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_toad_ocr_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   4,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_toad_ocr_proto_goTypes,
		DependencyIndexes: file_toad_ocr_proto_depIdxs,
		MessageInfos:      file_toad_ocr_proto_msgTypes,
	}.Build()
	File_toad_ocr_proto = out.File
	file_toad_ocr_proto_rawDesc = nil
	file_toad_ocr_proto_goTypes = nil
	file_toad_ocr_proto_depIdxs = nil
}

// Reference imports to suppress errors if they are not otherwise used.
var _ context.Context
var _ grpc.ClientConn

// This is a compile-time assertion to ensure that this generated file
// is compatible with the grpc package it is being compiled against.
const _ = grpc.SupportPackageIsVersion4

// ToadOcrClient is the client API for ToadOcr service.
//
// For semantics around ctx use and closing/ending streaming RPCs, please refer to https://godoc.org/google.golang.org/grpc#ClientConn.NewStream.
type ToadOcrClient interface {
	// Sends a greeting
	SayHello(ctx context.Context, in *HelloRequest, opts ...grpc.CallOption) (*HelloReply, error)
	Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictReply, error)
}

type toadOcrClient struct {
	cc grpc.ClientConn
}

func NewToadOcrClient(cc grpc.ClientConn) ToadOcrClient {
	return &toadOcrClient{cc}
}

func (c *toadOcrClient) SayHello(ctx context.Context, in *HelloRequest, opts ...grpc.CallOption) (*HelloReply, error) {
	out := new(HelloReply)
	err := c.cc.Invoke(ctx, "/idl.ToadOcr/SayHello", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

func (c *toadOcrClient) Predict(ctx context.Context, in *PredictRequest, opts ...grpc.CallOption) (*PredictReply, error) {
	out := new(PredictReply)
	err := c.cc.Invoke(ctx, "/idl.ToadOcr/Predict", in, out, opts...)
	if err != nil {
		return nil, err
	}
	return out, nil
}

// ToadOcrServer is the server API for ToadOcr service.
type ToadOcrServer interface {
	// Sends a greeting
	SayHello(context.Context, *HelloRequest) (*HelloReply, error)
	Predict(context.Context, *PredictRequest) (*PredictReply, error)
}

// UnimplementedToadOcrServer can be embedded to have forward compatible implementations.
type UnimplementedToadOcrServer struct {
}

func (*UnimplementedToadOcrServer) SayHello(context.Context, *HelloRequest) (*HelloReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method SayHello not implemented")
}
func (*UnimplementedToadOcrServer) Predict(context.Context, *PredictRequest) (*PredictReply, error) {
	return nil, status.Errorf(codes.Unimplemented, "method Predict not implemented")
}

func RegisterToadOcrServer(s *grpc.Server, srv ToadOcrServer) {
	s.RegisterService(&_ToadOcr_serviceDesc, srv)
}

func _ToadOcr_SayHello_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(HelloRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ToadOcrServer).SayHello(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/idl.ToadOcr/SayHello",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ToadOcrServer).SayHello(ctx, req.(*HelloRequest))
	}
	return interceptor(ctx, in, info, handler)
}

func _ToadOcr_Predict_Handler(srv interface{}, ctx context.Context, dec func(interface{}) error, interceptor grpc.UnaryServerInterceptor) (interface{}, error) {
	in := new(PredictRequest)
	if err := dec(in); err != nil {
		return nil, err
	}
	if interceptor == nil {
		return srv.(ToadOcrServer).Predict(ctx, in)
	}
	info := &grpc.UnaryServerInfo{
		Server:     srv,
		FullMethod: "/idl.ToadOcr/Predict",
	}
	handler := func(ctx context.Context, req interface{}) (interface{}, error) {
		return srv.(ToadOcrServer).Predict(ctx, req.(*PredictRequest))
	}
	return interceptor(ctx, in, info, handler)
}

var _ToadOcr_serviceDesc = grpc.ServiceDesc{
	ServiceName: "idl.ToadOcr",
	HandlerType: (*ToadOcrServer)(nil),
	Methods: []grpc.MethodDesc{
		{
			MethodName: "SayHello",
			Handler:    _ToadOcr_SayHello_Handler,
		},
		{
			MethodName: "Predict",
			Handler:    _ToadOcr_Predict_Handler,
		},
	},
	Streams:  []grpc.StreamDesc{},
	Metadata: "toad_ocr.proto",
}
