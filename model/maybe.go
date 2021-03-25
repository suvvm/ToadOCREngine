package model

import (
	"github.com/pkg/errors"
	"gorgonia.org/tensor"
	"suvvm.work/toad_ocr_engine/utils"
)

// Maybe 用于简化张量运算中频繁的错误处理，go的错误处理实在是太恶心了
//
// 参数
//	err	// 保存的错误信息
type Maybe struct {
	err error
}

// Do 执行方法，执行给定目标方法，并跟踪错误信息
//
// 入参
//	fn func() (tensor.Tensor, error)	// 目标方法
//
// 返回
//	tensor.Tensor						// 处理结果张量
func (m *Maybe) Do(fn func() (tensor.Tensor, error)) tensor.Tensor {
	if m.err != nil {	// 当前错误信息不为空
		return nil
	}
	var retVal tensor.Tensor
	// 执行目标方法
	if retVal, m.err = fn(); m.err == nil {
		return retVal
	}
	// 跟踪错误信息
	m.err = errors.WithStack(m.err)
	return nil
}

// Sigmoid Maybe简化后的神经网络激活函数
//
// 入参
//	data tensor.Tensor	// 数据张量
//
// 返回
//	tensor.Tensor	// 结果张量
func (m *Maybe) Sigmoid(data tensor.Tensor) tensor.Tensor {
	if m.err != nil {	// 当前错误信息不为空
		return nil
	}
	var retVal tensor.Tensor
	// 对张量中所有数据执行激活函数，结果数据写入retVal 原data不做改变
	if retVal, m.err = data.Apply(utils.Sigmoid); m.err == nil {
		return retVal
	}
	// 跟踪错误信息
	m.err = errors.WithStack(m.err)
	return nil
}
