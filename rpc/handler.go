package rpc

import (
	"bytes"
	"context"
	"flag"
	"github.com/otiai10/gosseract"
	"gorgonia.org/gorgonia"
	"image"
	"image/jpeg"
	"log"
	"os"
	"strconv"
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
	clientTs *gosseract.Client
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
	// lab, err = nn.SnnPredict(snn, in.Image)
	clientTs = gosseract.NewClient()
	// client.SetImageFromBytes(float64SliceAsByteSlice(in.Image))
	clientTs.SetPageSegMode(gosseract.PSM_SINGLE_CHAR)
	clientTs.SetVariable(gosseract.TESSEDIT_CHAR_WHITELIST,
		"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
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
	if in.NetFlag == common.CnnName {
		cnn.Lock.Lock()
		lab, err = nn.CnnPredict(cnn, in.Imagef)
		cnn.Lock.Unlock()
	} else {
		snn.Lock.Lock()
		val, ok := predictByTs(in)
		if !ok {
			lab, err = nn.SnnPredict(snn ,in.Imagef)
		} else {
			lab = strconv.Itoa(val)
		}
		snn.Lock.Unlock()
	}
	if err != nil {
		return &pb.PredictReply{Code: int32(*errorCode), Message: err.Error(), Label: *errorLab}, nil
	}
	return &pb.PredictReply{Code: int32(*successCode), Message: *successMsg, Label: lab}, nil
}

func predictByTs(in *pb.PredictRequest) (int, bool) {
	// defer client.Close()
	img, _, _ := image.Decode(bytes.NewReader(in.Image))
	out, err := os.Create("output/images/tmp.jpg")
	if err != nil {
		log.Fatal(err)
	}
	defer out.Close()
	jpeg.Encode(out, img, nil)
	err = clientTs.SetImage("output/images/tmp.jpg")
	if err != nil {
		log.Printf("set err:%v", err)
	}
	text, err := clientTs.Text()
	if err != nil {
		log.Printf("Text err:%v", err)
	}
	log.Printf("text:%v", text)
	if len(text) > 0 {
		val, ok := common.CharMap[text[0]]
		if !ok {
			return -1, false
		}
		return val, true
	}
	return -1, false
}
