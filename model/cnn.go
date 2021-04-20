package model

import (
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
	"io/ioutil"
	"log"
)

// CNN 四层卷积神经网络
// 卷积层表达式DropOut(MaxPoll(\sigma(w*x)))
// 退出与最大池化为同一层
type CNN struct {
	G	*gorgonia.ExprGraph				// 语法数学表达式图（可重用节点的AST）
	W0, W1, W2, W3, W4 *gorgonia.Node	// 各层权重
	D0, D1, D2, D3 float64				// 退出概率
	Out *gorgonia.Node
	OutVal gorgonia.Value
	VM gorgonia.VM
}

// fwd 前向传播函数
// 将图层包装在图层结构中执行每层激活
// 第0、1层，使用卷积神经网络的标准卷积stride =（1，1）和padding =（1，1）进行卷积
// 给定一个(N,1,28,28)的张量，卷积后得到一个(N,32,28,28)的张量
// 最大池化将图像长宽减半(2,2)后输出(M,32,14,14)
// 第3曾输出后将结果重构为一个秩为2的张量，使用softmax激活函数完成CNN实现
//
// 入参
//	x *gorgonia.Node	// 输入张量
//
// 返回
//	error				// 错误信息
func (cnn *CNN) Fwd (x *gorgonia.Node) error {
	// 第0层
	var c0, c1, c2, fc *gorgonia.Node
	var a0, a1, a2, a3 *gorgonia.Node
	var p0, p1, p2 *gorgonia.Node
	var l0, l1, l2, l3 *gorgonia.Node
	var err error

	log.Printf("x:%v", x)
	log.Printf("w0:%v", cnn.W0)

	if c0, err = gorgonia.Conv2d(x, cnn.W0, tensor.Shape{3, 3},
	[]int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {	// 卷积第0层
		return errors.Wrap(err, "Layer 0 Convolution failed")	// 跟踪错误信息
	}
	if a0, err = gorgonia.Rectify(c0); err != nil {	// 第0层激活
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	if p0, err = gorgonia.MaxPool2D(a0, tensor.Shape{2, 2}, 	// 第0层最大池化
	[]int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 0 Maxpooling failed")
	}
	if l0, err = gorgonia.Dropout(p0, cnn.D0); err != nil {	// 第0层概率退出
		return errors.Wrap(err, "Unable to apply a dropout to layer 0")
	}

	if c1, err = gorgonia.Conv2d(l0, cnn.W1, tensor.Shape{3, 3},	// 第1层卷积
	[]int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if a1, err = gorgonia.Rectify(c1); err != nil {	// 第1层激活
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	if p1, err = gorgonia.MaxPool2D(a1, tensor.Shape{2, 2}, 	// 第1层最大池化
	[]int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	if l1, err = gorgonia.Dropout(p1, cnn.D1); err != nil {		// 第1层概率退出
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}


	if c2, err = gorgonia.Conv2d(l1, cnn.W2, tensor.Shape{3, 3}, 	// 第2层卷积
	[]int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	if a2, err = gorgonia.Rectify(c2); err != nil {	// 第2层激活
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	if p2, err = gorgonia.MaxPool2D(a2, tensor.Shape{2, 2},	// 第2层最大池化
	[]int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 2 Maxpooling failed")
	}
	log.Printf("p2 shape %v", p2.Shape())
	// 修改第2层张量结构，将输出的秩为4的张量重新构造为秩为2的蟑螂
	var r2 *gorgonia.Node
	b, c, h, w := p2.Shape()[0], p2.Shape()[1], p2.Shape()[2], p2.Shape()[3]
	if r2, err = gorgonia.Reshape(p2, tensor.Shape{b, c * h * w}); err != nil {
		return errors.Wrap(err, "Unable to reshape layer 2")
	}
	log.Printf("r2 shape %v", r2.Shape())
	if l2, err = gorgonia.Dropout(r2, cnn.D2); err != nil {	// 第2层概率退出
		return errors.Wrap(err, "Unable to apply a dropout on layer 2")
	}
	_ = ioutil.WriteFile("output/tmp.dot", []byte(cnn.G.ToDot()), 0644)
	// 第3层
	if fc, err = gorgonia.Mul(l2, cnn.W3); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a3, err = gorgonia.Rectify(fc); err != nil {	// 第3层激活
		return errors.Wrapf(err, "Unable to activate fc")
	}
	if l3, err = gorgonia.Dropout(a3, cnn.D3); err != nil {	// 第3层概率退出
		return errors.Wrapf(err, "Unable to apply a dropout on layer 3")
	}
	// 输出解码
	var out *gorgonia.Node
	if out, err = gorgonia.Mul(l3, cnn.W4); err != nil {
		return errors.Wrapf(err, "Unable to multiply l3 and w4")
	}
	cnn.Out, err = gorgonia.SoftMax(out)	// 使用softmax 函数进行激活
	gorgonia.Read(cnn.Out, &cnn.OutVal)		// 保存预测结果
	return nil
}

func (cnn *CNN) Learnables() gorgonia.Nodes {
	return gorgonia.Nodes{cnn.W0, cnn.W1, cnn.W2, cnn.W3, cnn.W4}
}

func (cnn *CNN) VMBuild() gorgonia.VM {
	// 编译卷积神经网络构造表达式图并输出
	prog, locMap, _ := gorgonia.Compile(cnn.G)
	log.Printf("%v", prog)
	// 创建一个由构造表达式图编译为的VM
	vm := gorgonia.NewTapeMachine(cnn.G, gorgonia.WithPrecompiled(prog, locMap), gorgonia.BindDualValues(cnn.Learnables()...))
	cnn.VM = vm
	return vm
}
