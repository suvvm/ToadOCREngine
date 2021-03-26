package nn

import (
	"gorgonia.org/tensor"
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
