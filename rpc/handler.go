package rpc

import (
	"context"
	"flag"
	"github.com/otiai10/gosseract"
	"gorgonia.org/gorgonia"
	"log"
	"strconv"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/method"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/nn"
	pb "suvvm.work/toad_ocr_engine/rpc/idl"
	"suvvm.work/toad_ocr_engine/utils"
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
	//clientTs.SetLanguage("eng", "chi_sim")
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
	err := method.VerifySecret(ctx, in.AppId, in.BasicToken, in.Name)
	if err != nil {
		return &pb.HelloReply{Message: "Hello " + in.GetName()}, nil
	}
	log.Printf("Received: %v", in.GetName())
	return &pb.HelloReply{Message: "permission denied"}, nil
}

func (s *Server) Predict(ctx context.Context, in *pb.PredictRequest) (*pb.PredictReply, error) {
	log.Printf("app:%v Received Process from", in.AppId)
	err := method.VerifySecret(ctx, in.AppId, in.BasicToken, in.NetFlag + utils.PixelHashStr(in.Image))
	if err != nil {
		log.Printf("app:%v permission denied", in.AppId)
		return &pb.PredictReply{Code: int32(*errorCode), Message: "permission denied", Label: *errorLab}, nil
	}
	imgF64 := bytesToF64(in.Image)
	log.Printf("Predict %v", in.NetFlag)
	var lab string
	if in.NetFlag == common.CnnName {
		cnn.Lock.Lock()
		lab, err = nn.CnnPredict(cnn, imgF64)
		cnn.Lock.Unlock()
	} else {
		snn.Lock.Lock()
		val, ok := predictByTs(in.Image)
		if !ok {
			lab, err = nn.SnnPredict(snn, imgF64)
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

func predictByTs(imageData []byte) (int, bool) {
	// defer client.Close()
	//tmp := gocv.NewMat()
	//imageMat, err := gocv.NewMatFromBytes(28,28, 1, imageData)
	//if err != nil {
	//	log.Printf("NewMatFromBytes err:%v", err)
	//	return -1, false
	//}
	////gocv.BitwiseNot(imageMat, &imageMat)
	//gocv.IMWrite("output/images/tmp.png", imageMat)
	err := method.VisualizeBytes(imageData, "tmp.png")
	if err != nil {
		log.Printf("VisualizeBytesc err:%v", err)
		return -1, false
	}
	//err = clientTs.SetImageFromBytes(imageMat.ToBytes())
	err = clientTs.SetImage("output/images/tmp.png")
	if err != nil {
		log.Printf("SetImageFromBytes err:%v", err)
		return -1, false
	}
	// err = clientTs.SetImage("output/images/tmp.png")
	//if err != nil {
	//	log.Printf("set err:%v", err)
	//}
	text, err := clientTs.Text()
	if err != nil {
		log.Printf("Text err:%v", err)
		return -1, false
	}
	log.Printf("text:%v", text)
	if len(text) > 0 {
		val, ok := common.CharMap[text[0]]
		if !ok {
			log.Printf("clientTs resp not in map")
			return -1, false
		}
		return val, true
	}
	log.Printf("text <= 0")
	return -1, false
}

func bytesToF64(data []byte) []float64 {
	dataF64 := make([]float64, 0)
	for _, b := range data {
	   dataF64 = append(dataF64, utils.PixelWeight(b))
	}
	return dataF64
}
