package model

import (
	"errors"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf64"
)

// SNN 基础神经网络结构
// 输入层是common.MNISTRawImageRows * common.MNISTRawImageCols个float64型切片
// 前向馈入(矩阵惩罚后执行激活函数)形成隐层
// 隐层继续馈入形成输出层（MNIST为10个float64组成的向量，就是标签的热独编码）
//
// 参数
//	Hidden *tensor.Dense	// 输入层与隐层之间的权重矩阵
//	Final *tensor.Dense	// 隐层与输出层之间的权重矩阵
//	B0 float64				// 隐层偏置
//	B1 float64				// 输出层偏置
type SNN struct {
	Hidden *tensor.Dense
	Final *tensor.Dense
	B0 float64
	B1 float64
}

// Predict 前向传播函数
//
// 入参
//	data tensor.Tensor	// 输入参数张量
//
// 返回
//	int					// 输出层最大值索引
//	error				// 错误信息
func (snn *SNN) Predict(data tensor.Tensor) (int, error) {
	if data.Dims() != 1 {
		return -1, errors.New("Expected a vector")
	}
	var m Maybe
	hidden := m.Do(func() (tensor.Tensor, error) {
		return snn.Hidden.MatVecMul(data)	// 矩阵乘法 Hidden * data
	})
	hiddenSigmoid := m.Sigmoid(hidden)	// 执行激活函数形成隐层
	final := m.Do(func() (tensor.Tensor, error) {	// 矩阵乘法 Final * hiddenSigmoid
		return snn.Final.MatVecMul(hiddenSigmoid)
	})
	pred := m.Sigmoid(final)	// 执行激活函数得到输出层
	if m.err != nil {
		return -1, m.err
	}
	return vecf64.Argmax(pred.Data().([]float64)), nil
}
