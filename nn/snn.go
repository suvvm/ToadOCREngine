package nn

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
	"log"
	"suvvm.work/toad_ocr_engine/common"
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
//	dataZCA tensor.Tensor	// zca白化后的图像张量
//	datalabs tensor.Tensor	// 标签张量
func SNNTrainingTest(dataZCA, datalabs tensor.Tensor) {
	log.Printf("SNN Start Training")
	// 获取张量底层float64数组切片
	nat, err := native.MatrixF64(dataZCA.(*tensor.Dense))
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	// 构建一个三层基础神经网络
	// 输入层神经元common.MNISTRawImageRows * common.MNISTRawImageCols个
	// 隐层神经元100个
	// 输出层神经元10个
	snn := NewSNN(common.MNISTRawImageRows * common.MNISTRawImageCols, 100, 10)
	// 构造成本数组
	costs := make([]float64, 0, dataZCA.Shape()[0])
	// 训练5组样本
	for i := 0; i < 5; i++ {
		dataZCAShape := dataZCA.Shape()
		var image, label tensor.Tensor
		var err error
		for j := 0; j < dataZCAShape[0]; j++ {
			if image, err = dataZCA.Slice(model.MakeRS(j, j + 1)); err != nil {
				log.Fatalf("Unable to slice image %d", j)
			}
			if label, err = datalabs.Slice(model.MakeRS(j, j + 1)); err != nil {
				log.Fatalf("Unable to slice label %d", j)
			}
			var cost float64
			if cost, err = snn.SNNTrain(image, label, 0.1); err != nil {
				log.Fatalf("Training error:%v", err)
			}
			costs = append(costs, cost)

		}
		log.Printf("%d\tcosts:%v", i, utils.Avg(costs))
		utils.ShuffleX(nat)
		costs = costs[:0]
	}
	log.Printf("End Training!")
}
