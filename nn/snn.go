package nn

import (
	"fmt"
	"github.com/cheggaaa/pb"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
	"gorgonia.org/vecf64"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/utils"
	"time"
)

// NewSNN 构建新的基础神经网络
//
// 入参
//	input int	// 输入数据长度
//	hidden int	// 隐层神经元数量
//	output int	// 输出层神经元数量
//
// 返回
//	*model.SNN	// 基础神经网络
func NewSNN(input, hidden, output int) *model.SNN {
	hiddenData := make([]float64, hidden * input)	// 隐层权重矩阵数据切片
	finalData := make([]float64, hidden * output)	// 输出层权重矩阵数据切片
	// 随机填充
	utils.FillRandom(hiddenData, float64(len(hiddenData)))
	utils.FillRandom(finalData, float64(len(finalData)))
	// 创建张量
	hiddenT := tensor.New(tensor.WithShape(hidden, input), tensor.WithBacking(hiddenData))
	finalT := tensor.New(tensor.WithShape(output, hidden), tensor.WithBacking(finalData))
	// 返回新的基础神经网络
	return &model.SNN{
		Hidden: hiddenT,
		Final: finalT,
	}
}

// SNNTrainingTest 基础神经网络训练测试
// 构建一个基础神经网络并训练5组数据
//
// 入参
//	dataImgs tensor.Tensor	// 原始图像张量
//	dataZCA tensor.Tensor	// zca白化后的图像张量
//	datalabs tensor.Tensor	// 标签张量
func SNNTrainingTest(dataImgs ,dataZCA, datalabs tensor.Tensor) {
	// 构建一个三层基础神经网络
	// 输入层神经元common.RawImageRows * common.RawImageCols个
	// 隐层神经元common.RawImageRows * common.RawImageCols个
	// 输出层神经元10个
	snn := NewSNN(common.RawImageRows* common.RawImageCols,
		common.RawImageRows * common.RawImageCols, common.EMNISTByClassNumLabels)
	SNNTraining(snn, dataImgs, dataZCA, datalabs, 5)
}

// SNNTraining 基础神经网络训练
// 构建一个基础神经网络并训练5组数据
//
// 入参
//	snn model.SNN			// 基础神经网络
//	dataImgs tensor.Tensor	// 原始图像张量
//	dataZCA tensor.Tensor	// zca白化后的图像张量
//	datalabs tensor.Tensor	// 标签张量
//	epochNum int			// 训练次数
func SNNTraining(snn *model.SNN, dataImgs ,dataZCA, datalabs tensor.Tensor, epochNum int) {
	log.Printf("SNN Start Training")
	// 获取张量底层float64数组切片
	nat, err := native.MatrixF64(dataZCA.(*tensor.Dense))
	if err != nil {
		log.Fatalf("err:%v", err)
	}

	tmpDataSize := 50000


	// 构造成本数组
	costs := make([]float64, 0, dataZCA.Shape()[0])
	// 训练基础神经网络epochNum次
	// bar := pb.New(dataZCA.Shape()[0])
	bar := pb.New(tmpDataSize)
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(common.BarMaxWidth)
	for i := 0; i < epochNum; i++ {
		bar.Prefix(fmt.Sprintf("Epoch %d", i))
		bar.Set(0)
		bar.Start()
		// dataZCAShape := dataZCA.Shape()
		var image, label tensor.Tensor
		var err error
		// for j := 0; j < dataZCAShape[0]; j++ {
		for j := 0; j < tmpDataSize; j++ {
			if image, err = dataImgs.Slice(model.MakeRS(j, j + 1)); err != nil {
				log.Fatalf("Unable to slice image %d", j)
			}
			if label, err = datalabs.Slice(model.MakeRS(j, j + 1)); err != nil {
				log.Fatalf("Unable to slice label %d", j)
			}
			var cost float64
			if cost, err = snn.Train(image, label, 0.1); err != nil {
				log.Fatalf("Training error:%v", err)
			}
			costs = append(costs, cost)
			bar.Increment()
		}
		log.Printf("epoch:%d\tcosts:%v", i, utils.Avg(costs))
		utils.ShuffleX(nat)
		costs = costs[:0]
		snn.TrainEpoch += 1
	}
	bar.Finish()
	if err = snn.Persistence(); err != nil {
		log.Fatalf("Save snn weights err:%v", err)
	}
	log.Printf("End Training!")
}

// SNNTesting 基础神经网络测试
// 在已有神经网络基础上运行测试集
//
// 入参
//	snn model.SNN			// 训练完成的基础神经网络
//	dataImgs tensor.Tensor	// 测试图像张量
//	datalabs tensor.Tensor	// 测试图像标签
func SNNTesting(snn *model.SNN, dataImgs, datalabs tensor.Tensor) {
	// shape := dataImgs.Shape()
	var err error
	var correct, total float64
	var image, label tensor.Tensor
	var predicted, errCnt int
	tmpDataSize := 50000
	bar := pb.New(tmpDataSize)
	// bar := pb.New(shape[0])
	bar.SetRefreshRate(time.Second)
	bar.SetMaxWidth(common.BarMaxWidth)
	bar.Prefix("Testing")
	bar.Set(0)
	bar.Start()
	for i := 0; i < tmpDataSize; i++ {
	// for i := 0; i < shape[0]; i++ {
		if image, err = dataImgs.Slice(model.MakeRS(i, i + 1)); err != nil {
			log.Fatalf("Unable to slice image %d", i)
		}
		if label, err = datalabs.Slice(model.MakeRS(i, i+1)); err != nil {
			log.Fatalf("Unable to slice label %d", i)
		}

		label := vecf64.Argmax(label.Data().([]float64))
		if predicted, err = snn.Predict(image); err != nil {
			log.Fatalf("Failed to predict %d", i)
		}

		if predicted == label {
			correct++
		} else {
			//if err = method.Visualize(image, 1, 1,
			//	fmt.Sprintf("%d_label:%d_predicted:%d.png", i, label, predicted)); err != nil {
			//	log.Fatalf("visualize error:%v", err)
			//}
			errCnt++
		}
		total++
		bar.Increment()
	}
	bar.Finish()
	log.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
}

func RunSNN() {
	dataImgs, datalabs, testData, testLbl := utils.LoadNIST(common.EMNSITByClassTrainImagesPath,
		common.EMNISTByClassTrainLabelsPath, common.EMNISTByClassTestImagesPath, common.EMNISTByClassTestLabelsPath)
	// 对图像进行ZCA白化
	dataZCA, err := utils.ZCA(dataImgs)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	_, err = os.Stat("snn_weights")
	if err != nil && !os.IsExist(err){
		//// 可视化一个由10 * 10个原始图像组成的图像
		//if err = method.Visualize(dataImgs, common.MNISTUnitRows, common.MNISTUnitCols, "image.png"); err != nil {
		//	log.Fatalf("visualize error:%v", err)
		//}
		//// 可视化上方图像的ZCA白化版本
		//if err = method.Visualize(dataZCA, common.MNISTUnitRows, common.MNISTUnitCols, "imageZCA.png"); err != nil {
		//	log.Fatalf("visualize error:%v", err)
		//}
		// 构建一个三层基础神经网络
		// 输入层神经元common.RawImageRows * common.MNISTRawImageCols个
		// 隐层神经元common.RawImageRows * common.MNISTRawImageCols个
		// 输出层神经元10个
		snn := NewSNN(common.RawImageRows* common.RawImageCols,
			common.RawImageRows * common.RawImageCols, common.EMNISTByClassNumLabels)
		//snn := nn.NewSNN(common.RawImageRows * common.RawImageCols,
		//	common.RawImageRows * common.RawImageCols, 10)
		// 训练SNN10次
		SNNTraining(snn, dataImgs, dataZCA, datalabs, common.SNNEpoch)
	}
	snn, err := model.LoadSNNFromSave()
	if err != nil {
		log.Fatalf("Failed at load snn weights %v", err)
	}
	SNNTraining(snn, dataImgs, dataZCA, datalabs, common.SNNEpoch)
	SNNTesting(snn, testData, testLbl)
}