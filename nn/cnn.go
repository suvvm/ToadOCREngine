package nn

import (
	"fmt"
	"github.com/cheggaaa/pb"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf64"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
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

// CNNTraining 卷积神经网络训练函数，根据传入的图像与标签信息对现有cnn进行训练
//
// 入参
//	cnn *model.CNN			// 卷积神经网络
//	dataImgs tensor.Tensor	// 训练集图像张量
//	dataLabs tensor.Tensor	// 训练集标签张量
func CNNTraining(cnn *model.CNN, dataImgs, dataLabs tensor.Tensor) {
	dataSize := dataImgs.Shape()[0]
	// 表达式网络输入数据x，内容为等候训练或预测的图像数据张量
	x := gorgonia.NewTensor(cnn.G, tensor.Float64, 4, gorgonia.WithShape(common.CNNBatchSize,
		common.MNISTRawImageChannel, common.MNISTRawImageRows,common.MNISTRawImageCols), gorgonia.WithName("x"))
	// 表达式网络输入数据y，内容为上述图像数据对应标签张量
	y := gorgonia.NewMatrix(cnn.G, tensor.Float64, gorgonia.WithShape(common.CNNBatchSize,
		common.MNISTNumLabels), gorgonia.WithName("y"))
	var err error
	// 创建求解器
	solver := gorgonia.NewRMSPropSolver(gorgonia.WithBatchSize(float64(common.CNNBatchSize)))
	// 计算每个百分比所占数据量大小
	batches := dataSize / common.CNNBatchSize
	// 构造进度条
	// 设置刷新率，每秒刷新进度条
	// 设置进度条最大宽度
	bar := pb.New(batches)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(common.BarMaxWidth)
	// 开始分阶段训练，每个阶段训练一遍全部数据集合
	for i := 0; i < common.CNNEpoch; i++ {
		// 重制进度条与本阶段准确率信息
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		accuracyGuess := float64(0)
		// 开始分批次训练数据，每个批次训练common.CNNBatchSize个原始图像
		for b := 0; b < batches; b++ {
			start := b * common.CNNBatchSize
			end := start + common.CNNBatchSize
			if start >= dataSize {
				break
			}
			if end > dataSize {
				end = dataSize
			}
			// 切片图像与标签张量
			var xVal, yVal tensor.Tensor
			if xVal, err = dataImgs.Slice(model.MakeRS(start, end)); err != nil {
				log.Fatal("Unable to slice x")
			}
			if yVal, err = dataLabs.Slice(model.MakeRS(start, end)); err != nil {
				log.Fatal("Unable to slice y")
			}
			if err = xVal.(*tensor.Dense).Reshape(common.CNNBatchSize, common.MNISTRawImageChannel,
				common.MNISTRawImageRows, common.MNISTRawImageCols); err != nil {
				log.Fatalf("Unable to reshape %v", err)
			}
			gorgonia.Let(x, xVal)
			gorgonia.Let(y, yVal)
			if err = cnn.VM.RunAll(); err != nil {	// 运行表达式网络
				log.Fatalf("Failed at epoch  %d: %v", i, err)
			}
			outputData := cnn.OutVal.Data().([]float64)	// 获取预测信息
			outputLab := tensor.New(tensor.WithShape(common.CNNBatchSize, common.MNISTNumLabels), tensor.WithBacking(outputData))
			// 准确率统计
			for j := 0; j < yVal.Shape()[0]; j++ {
				yRowT, _ :=  yVal.Slice(model.MakeRS(j, j + 1))
				yRow := yRowT.Data().([]float64)
				lab := vecf64.Argmax(yRow)
				preRowT, _ := outputLab.Slice(model.MakeRS(j, j + 1))
				preRow := preRowT.Data().([]float64)
				prelab :=  vecf64.Argmax(preRow)
				if lab == prelab {
					accuracyGuess += 1.0 / float64(dataSize)
				}
			}
			// RMSProp梯度下降学习
			if err = solver.Step(gorgonia.NodesToValueGrads(cnn.Learnables())); err != nil {
				log.Fatalf("Failed at solver %v", err)
			}
			cnn.VM.Reset()
			bar.Increment()
		}
		// 输出本次训练信息
		log.Printf("Epoch %d | guess %v", i, accuracyGuess)
	}
	if err := cnn.Persistence(); err != nil {
		log.Fatalf("Failed at persistence %v", err)
	}
	bar.Finish()
}

// CNNTesting 卷积神经网络测试函数，根据传入的测试集图像与标签信息对现有cnn进行测试
//
// 入参
//	cnn *model.CNN			// 现有卷积神经网络
//	testData tensor.Tensor	// 测试集图像张量
//	testLbl tensor.Tensor	// 测试集标签张量
func CNNTesting(cnn *model.CNN, testData, testLbl tensor.Tensor) {
	var out gorgonia.Value
	rv := gorgonia.Read(cnn.Out, &out)
	fwdNet := cnn.G.SubgraphRoots(rv)
	vm := gorgonia.NewTapeMachine(fwdNet)
	cnn.VM = vm
	var correct, total, errcount float64
	var testBs int
	var err error
	testBs = 100	// 测试集每批次测试数量
	// 表达式网络输入数据x，内容为等候训练或预测的图像数据张量
	x := gorgonia.NewTensor(cnn.G, tensor.Float64, 4, gorgonia.WithShape(testBs, common.MNISTRawImageChannel,
		common.MNISTRawImageRows, common.MNISTRawImageCols), gorgonia.WithName("x"))
	shape := testData.Shape()
	testDataSize := shape[0]
	batches := testDataSize / testBs
	// 构造进度条
	bar := pb.New(batches)
	// 设置刷新率，每秒刷新进度条
	bar.SetRefreshRate(time.Second)
	// 进度条最大宽度
	bar.SetMaxWidth(common.BarMaxWidth)
	bar.Prefix("Testing")
	bar.Set(0)
	bar.Start()
	// 开始分批测试
	for b := 0; b < batches; b++ {
		start := b * testBs
		end := start + testBs
		if start >= testDataSize {
			break
		}
		if end > testDataSize {
			end = testDataSize
		}
		// 测试集张量切片
		var xVal, yVal tensor.Tensor
		if xVal, err = testData.Slice(model.MakeRS(start, end)); err != nil {
			log.Fatal("Unable to slice x")
		}
		if yVal, err = testLbl.Slice(model.MakeRS(start, end)); err != nil {
			log.Fatal("Unable to slice y")
		}
		if err = xVal.(*tensor.Dense).Reshape(testBs, common.MNISTRawImageChannel,
			common.MNISTRawImageRows, common.MNISTRawImageCols); err != nil {
			log.Fatalf("Unable to reshape %v", err)
		}
		gorgonia.Let(x, xVal)
		if err = cnn.VM.RunAll(); err != nil {	// 运行表达式网络
			log.Fatalf("Failed at epoch  %d: %v", b, err)
		}
		// 记录测试结果
		label, _ := yVal.(*tensor.Dense).Argmax(1)
		predicted, _ := cnn.OutVal.(*tensor.Dense).Argmax(1)
		lblData := label.Data().([]int)
		// log.Printf("\np:%v\n",  predicted.Data().([]int))
		for j, p := range predicted.Data().([]int) {
			if p == lblData[j] {
				correct++
			} else {
				errcount++
			}
			total++
		}
		cnn.VM.Reset()
		bar.Increment()
	}
	bar.Finish()
	// log.Printf("Error/Totals: %v/%v = %1.3f\n", errcount, total, errcount/total)
	log.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
}

// RunCNN 卷积神经网络运行函数
func RunCNN() {
	dataImgs, datalabs, testData, testLbl := utils.LoadMNIST()
	//dataZCA, err := utils.ZCA(dataImgs)
	//if err != nil {
	//	log.Fatalf("err:%v", err)
	//}
	_, err := os.Stat("cnn_weights")
	if err != nil && !os.IsExist(err){
		cnnImgsData := dataImgs.Shape()[0]
		if err := dataImgs.Reshape(cnnImgsData, common.MNISTRawImageChannel,
			common.MNISTRawImageRows, common.MNISTRawImageCols); err != nil {
			log.Fatal(err)
		}
		// 构造表达式图
		g := gorgonia.NewGraph()
		x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(common.CNNBatchSize,
			common.MNISTRawImageChannel, common.MNISTRawImageRows,
			common.MNISTRawImageCols), gorgonia.WithName("x"))
		// 表达式网络输入数据y，内容为上述图像数据对应标签张量
		y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(common.CNNBatchSize,
			common.MNISTNumLabels), gorgonia.WithName("y"))
		// 构建卷积神经网络
		cnn := NewCNN(g)
		if err := cnn.Fwd(x); err != nil {
			log.Fatalf("%+v", err)
		}
		losses := gorgonia.Must(gorgonia.HadamardProd(cnn.Out, y))
		cost := gorgonia.Must(gorgonia.Mean(losses))
		cost = gorgonia.Must(gorgonia.Neg(cost))
		// 跟踪成本
		var costVal gorgonia.Value
		gorgonia.Read(cost, &costVal)
		// 根据权重矩阵获取成本
		if _, err := gorgonia.Grad(cost, cnn.Learnables()...); err != nil {
			log.Fatal(err)
		}
		cnn.VMBuild()
		defer cnn.VM.Close()
		CNNTraining(cnn, dataImgs, datalabs)
	}
	// testData, testLbl, _, _ := utils.LoadMNIST()
	// 对图像进行ZCA白化

	//nat, err := native.MatrixF64(dataZCA.(*tensor.Dense))
	//if err != nil {
	//	log.Fatal(err)
	//}
	// 使用卷积神经网络前将图像数据转换为(cnnImgsData, 1, 28, 28) (BHCW)
	cnn, err := model.LoadCNNFromSave()
	defer cnn.VM.Close()
	CNNTraining(cnn, dataImgs, datalabs)
	if err != nil {
		log.Fatalf("Unable to load cnn file %v", err)
	}
	// cnnLoaded.VMBuild()
	CNNTesting(cnn, testData, testLbl)
}
