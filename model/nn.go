package model

import "gorgonia.org/tensor"

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
