package nn

import (
	"fmt"
	"github.com/cheggaaa/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/method"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/utils"
	"time"
)

// NewCNN 新建卷积神经网络
// w0 第0层权重(32, 1, 3, 3) 批次数（滤波器数量）32 颜色通道1（灰度图） 滤波器大小3 * 3 （为啥 3 * 3， 因为劳资愿意）
// 卷积层改变了输入形状下方第1层与第2层的形状要根据上一层输出进行修改
// w1 第1层权重(64, 32, 3, 3)
// w2 第2层权重(128, 64, 3, 3)
// 第2层将输出一个秩为4的张量，为了方便执行乘法在第3层将其重构为一个秩为2的张量
// w3 第3层权重(128 * 3 * 3, 625)
// 第4层进行输出解码得到结果概率
// w4 第4层权重(625, 10)
// d0～d3前三层20%的激活随机归零,最后一层55%的激活随机归零
//
// 入参
//	g *gorgonia.ExprGraph	// 语法数学表达式图(可重用节点的AST)
//
// 返回
//	*model.CNN				// 卷积神经网络
func NewCNN(g *gorgonia.ExprGraph) *model.CNN {
	// w0层权重
	w0 := gorgonia.NewTensor(g, tensor.Float64, 4,
		gorgonia.WithShape(32, 1, 3, 3), gorgonia.WithName("w0"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	// w1层权重
	w1 := gorgonia.NewTensor(g, tensor.Float64, 4,
		gorgonia.WithShape(64, 32, 3, 3), gorgonia.WithName("w1"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w2 := gorgonia.NewTensor(g, tensor.Float64, 4,
		gorgonia.WithShape(128, 64, 3, 3), gorgonia.WithName("w2"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w3 := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(128 * 3 * 3, 625), gorgonia.WithName("w3"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	w4 := gorgonia.NewMatrix(g, tensor.Float64,
		gorgonia.WithShape(625, 10), gorgonia.WithName("w4"),
		gorgonia.WithInit(gorgonia.GlorotN(1.0)))
	return &model.CNN{
		G: g,
		W0: w0,
		W1: w1,
		W2: w2,
		W3: w3,
		W4: w4,
		D0: 0.2,
		D1: 0.2,
		D2: 0.2,
		D3: 0.55,
	}
}

func RunCNN() {
	// 读取图像文件
	reader, err := os.Open("resources/mnist/train-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	images, err := utils.ReadImageFile(reader)
	// 读取标签文件
	reader, err = os.Open("resources/mnist/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	labels, err := utils.ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	// 打印图像与标签数量
	log.Printf("number of imgs:%d, number of labs:%d", len(images), len(labels))
	// 将图像与标签转化为张量
	dataImgs := method.PrepareX(images)
	datalabs := method.PrepareY(labels)
	// 对图像进行ZCA白化
	//dataZCA, err := utils.ZCA(dataImgs)
	//if err != nil {
	//	log.Fatalf("err:%v", err)
	//}
	//nat, err := native.MatrixF64(dataZCA.(*tensor.Dense))
	//if err != nil {
	//	log.Fatal(err)
	//}
	// 使用卷积神经网络前将图像数据转换为(cnnImgsData, 1, 28, 28) (BHCW)
	cnnImgsData := dataImgs.Shape()[0]
	bs := 100
	if err := dataImgs.Reshape(cnnImgsData, 1, 28, 28); err != nil {
		log.Fatal(err)
	}
	var dt tensor.Dtype
	dt =  tensor.Float64
	// 构造表达式图
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, dt, 4, gorgonia.WithShape(bs, 1, 28, 28), gorgonia.WithName("x"))
	y := gorgonia.NewMatrix(g, dt, gorgonia.WithShape(bs, 10), gorgonia.WithName("y"))
	log.Printf("outx:%v",x)
	log.Printf("outy:%v",y)
	// 构建卷积神经网络
	m := NewCNN(g)
	if err = m.Fwd(x); err != nil {
		log.Fatalf("%+v", err)
	}
	losses := gorgonia.Must(gorgonia.HadamardProd(m.Out, y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))
	// 跟踪成本
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)
	// 根据权重矩阵获取成本
	if _, err = gorgonia.Grad(cost, m.Learnables()...); err != nil {
		log.Fatal(err)
	}
	// 编译卷积神经网络构造表达式图并输出
	prog, locMap, _ := gorgonia.Compile(g)
	log.Printf("%v", prog)
	// 创建一个由构造表达式图编译为的VM
	vm := gorgonia.NewTapeMachine(g, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(m.Learnables()...))
	// 创建求解器
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(bs)))
	defer vm.Close()
	// 计算每个百分比所占数据量大小
	batches := cnnImgsData / bs
	// log.Printf("Batches %d", batches)
	// 构造进度条
	bar := pb.New(batches)
	// 设置刷新率，每秒刷新进度条
	bar.SetRefreshRate(time.Second)
	// 进度条最大宽度
	bar.SetMaxWidth(80)

	for i := 0; i < 100; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		for b := 0; b < batches; b++ {
			start := b * bs
			end := start + bs
			if start >= cnnImgsData {
				break
			}
			if end > cnnImgsData {
				end = cnnImgsData
			}
			var xVal, yVal tensor.Tensor
			if xVal, err = dataImgs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice x")
			}

			if yVal, err = datalabs.Slice(sli{start, end}); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(bs, 1, 28, 28); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}

			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = vm.RunAll(); err != nil {
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			solver.Step(gorgonia.NodesToValueGrads(m.Learnables()))
			vm.Reset()
			bar.Increment()
		}
		log.Printf("Epoch %d | cost %v", i, costVal)
	}

	var out gorgonia.Value
	rv := gorgonia.Read(m.Out, &out)
	fwdNet := m.G.SubgraphRoots(rv)
	vm2 := gorgonia.NewTapeMachine(fwdNet)

	reader, err = os.Open("resources/mnist/t10k-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testImgs, err := utils.ReadImageFile(reader)
	if err != nil {
		log.Fatal(err)
	}
	reader, err = os.Open("resources/mnist/t10k-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testlabels, err := utils.ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}

	testData := method.PrepareX(testImgs)
	testLbl := method.PrepareY(testlabels)
	shape := testData.Shape()
	//testData2, err := utils.ZCA(testData)
	//if err != nil {
	//	log.Fatal(err)
	//}

	var correct, total float64
	var oneimg, onelabel tensor.Tensor
	var predicted, errcount int
	for i := 0; i < shape[0]; i++ {
		if oneimg, err = testData.Slice(model.MakeRS(i, i+1)); err != nil {
			log.Fatalf("Unable to slice one image %d", i)
		}
		if onelabel, err = testLbl.Slice(model.MakeRS(i, i+1)); err != nil {
			log.Fatalf("Unable to slice one label %d", i)
		}
		label := utils.Argmax(onelabel.Data().([]float64))
		gorgonia.Let(x, oneimg)
		if err = vm2.RunAll(); err != nil {
			log.Fatal("Predicting %d failed %v", i, err)
		}
		outRaw := out.Data().([]float64)
		predicted = utils.Argmax(outRaw)

		if predicted == label {
			correct++
		} else if errcount < 5 {
			method.Visualize(oneimg, 1, 1, fmt.Sprintf("%d_%d_%d.png", i, label, predicted))
			errcount++
		}
		total++
	}
	fmt.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
	log.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
}

type sli struct {
	start, end int
}

func (s sli) Start() int { return s.start }
func (s sli) End() int   { return s.end }
func (s sli) Step() int  { return 1 }
