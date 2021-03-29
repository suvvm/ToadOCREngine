package nn

import (
	"fmt"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
	"gorgonia.org/vecf64"
	"log"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/method"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/utils"
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
	// 输入层神经元common.MNISTRawImageRows * common.MNISTRawImageCols个
	// 隐层神经元common.MNISTRawImageRows * common.MNISTRawImageCols个
	// 输出层神经元10个
	snn := NewSNN(common.MNISTRawImageRows * common.MNISTRawImageCols,
		common.MNISTRawImageRows * common.MNISTRawImageCols, 10)
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
	// 构造成本数组
	costs := make([]float64, 0, dataZCA.Shape()[0])
	// 训练基础神经网络epochNum次
	for i := 0; i < epochNum; i++ {
		dataZCAShape := dataZCA.Shape()
		var image, label tensor.Tensor
		var err error
		for j := 0; j < dataZCAShape[0]; j++ {
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

		}
		log.Printf("epoch:%d\tcosts:%v", i, utils.Avg(costs))
		utils.ShuffleX(nat)
		costs = costs[:0]
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
	shape := dataImgs.Shape()
	var err error
	var correct, total float64
	var image, label tensor.Tensor
	var predicted, errCnt int
	for i := 0; i < shape[0]; i++ {
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
			if err = method.Visualize(image, 1, 1,
				fmt.Sprintf("%d_label:%d_predicted:%d.png", i, label, predicted)); err != nil {
				log.Fatalf("visualize error:%v", err)
			}
			errCnt++
		}
		total++
	}
	log.Printf("Correct/Totals: %v/%v = %1.3f\n", correct, total, correct/total)
}
