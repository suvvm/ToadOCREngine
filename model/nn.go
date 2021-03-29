package model

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"gorgonia.org/vecf64"
	"suvvm.work/toad_ocr_engine/utils"
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
		return snn.Hidden.MatVecMul(data)	// 矩阵向量乘法 Hidden × data
	})
	hiddenSigmoid := m.Sigmoid(hidden)	// 执行激活函数形成隐层权重矩阵
	final := m.Do(func() (tensor.Tensor, error) {	// 矩阵向量乘法 Final × hiddenSigmoid
		return snn.Final.MatVecMul(hiddenSigmoid)
	})
	pred := m.Sigmoid(final)	// 执行激活函数得到输出层权重矩阵
	if m.err != nil {
		return -1, m.err
	}
	return vecf64.Argmax(pred.Data().([]float64)), nil
}

// SNNTrain 基础神经网络单次训练函数
// 进行单次训练并得到本次训练的成本（误差平方和对预测结果求偏导数）
//
// 入参
//	x tensor.Tensor		// 图像张量
//	y tensor.Tensor		// 标签张量
//	learnRate float64	// 梯度更新值（学习速率）
// 返回
//	float64				// 本次成本
//	error				// 错误信息
func (snn *SNN) Train(x, y tensor.Tensor, learnRate float64) (float64, error) {
	// 预测
	var m Maybe
	// 调整图像张量为一维张量
	m.Do(func() (tensor.Tensor, error) {
		err := x.Reshape(x.Shape()[0], 1)
		return x, err
	})
	// 调整标签张量为一维张量
	m.Do(func() (tensor.Tensor, error) {
		err := y.Reshape(10, 1)
		return y, err
	})
	// 隐层权重矩阵乘输入数据
	hidden := m.Do(func() (tensor.Tensor, error) {
		return tensor.MatMul(snn.Hidden, x)
	})
	// 执行激活函数
	hiddenSigmoid :=  m.Sigmoid(hidden)

	//hiddenSigmoid := m.Do(func() (tensor.Tensor, error) {
	//	return hidden.Apply(utils.Sigmoid, tensor.UseUnsafe())
	//})

	// 隐层输出层权重矩阵乘隐层数据矩阵
	final := m.Do(func() (tensor.Tensor, error) {
		return tensor.MatMul(snn.Final, hiddenSigmoid)
	})
	pred := m.Sigmoid(final)
	//pred := m.Do(func() (tensor.Tensor, error) {
	//	return final.Apply(utils.Sigmoid, tensor.UseUnsafe())
	//})
	// 反向传播
	var cost float64
	// 计算输出层误差， 标签减去结果得到误差
	outputErrors := m.Do(func() (tensor.Tensor, error) {
		return tensor.Sub(y, pred)
	})
	cost = utils.Sum(outputErrors.Data().([]float64))
	// 计算隐层误差
	hidErrs := m.Do(func() (tensor.Tensor, error) {
		// Final 转置
		if err := snn.Final.T(); err != nil {
			return nil, err
		}
		defer snn.Final.UT()	// 函数结束后取消转置
		// Final 转置乘输出层误差
		return tensor.MatMul(snn.Final, outputErrors)
	})
	if m.err != nil {
		return 0, m.err
	}
	// 结果矩阵的导数
	dpred := m.Do(func() (tensor.Tensor, error) {
		return pred.Apply(utils.DSigmoid, tensor.UseUnsafe())
	})
	// 元素乘法计算误差对结果矩阵的导数
	m.Do(func() (tensor.Tensor, error) {
		return tensor.Mul(dpred, outputErrors, tensor.UseUnsafe())
	})
	// 误差对隐层输出层权重矩阵的导数
	dpred_dfinal := m.Do(func() (tensor.Tensor, error) {
		if err := hiddenSigmoid.T(); err != nil {
			return nil, err
		}
		defer hiddenSigmoid.UT()
		// 误差乘输入层数据的转置
		return tensor.MatMul(outputErrors, hiddenSigmoid)
	})
	// 输入矩阵的导数
	dHiddenSigmoid := m.Do(func() (tensor.Tensor, error) {
		return hiddenSigmoid.Apply(utils.DSigmoid)
	})
	// 误差乘输入矩阵的导数
	m.Do(func() (tensor.Tensor, error) {
		return tensor.Mul(hidErrs, dHiddenSigmoid, tensor.UseUnsafe())
	})
	// 调整误差乘输入矩阵的导数张量为一维张量
	m.Do(func() (tensor.Tensor, error) {
		err := hidErrs.Reshape(hidErrs.Shape()[0], 1)
		return hidErrs, err
	})
	// 误差对输入层隐层权重矩阵的导数
	dcost_dhidden := m.Do(func() (tensor.Tensor, error) {
		if err := x.T(); err != nil {
			return nil, err
		}
		defer x.UT()
		return tensor.MatMul(hidErrs, x)
	})
	// 使用上述求得的导数作为梯度，更新输入矩阵
	m.Do(func() (tensor.Tensor, error) {
		return tensor.Mul(dpred_dfinal, learnRate, tensor.UseUnsafe())
	})
	m.Do(func() (tensor.Tensor, error) {
		return tensor.Mul(dcost_dhidden, learnRate, tensor.UseUnsafe())
	})
	m.Do(func() (tensor.Tensor, error) {
		return tensor.Add(snn.Final, dpred_dfinal, tensor.UseUnsafe())
	})
	m.Do(func() (tensor.Tensor, error) {
		return tensor.Add(snn.Hidden, dcost_dhidden, tensor.UseUnsafe())
	})
	return cost, m.err
}
