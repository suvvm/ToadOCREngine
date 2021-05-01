package model

import (
	"encoding/gob"
	"github.com/pkg/errors"
	"gorgonia.org/gorgonia"
	nnops "gorgonia.org/gorgonia/ops/nn"
	"gorgonia.org/tensor"
	"io/ioutil"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
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
	TrainEpoch int
}

// Fwd 前向传播函数
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

	if c0, err = nnops.Conv2d(x, cnn.W0, tensor.Shape{3, 3},
	[]int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {	// 卷积第0层
		return errors.Wrap(err, "Layer 0 Convolution failed")	// 跟踪错误信息
	}
	if a0, err = nnops.Rectify(c0); err != nil {	// 第0层激活
		return errors.Wrap(err, "Layer 0 activation failed")
	}
	if p0, err = nnops.MaxPool2D(a0, tensor.Shape{2, 2}, 	// 第0层最大池化
	[]int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 0 Maxpooling failed")
	}
	if l0, err = nnops.Dropout(p0, cnn.D0); err != nil {	// 第0层概率退出
		return errors.Wrap(err, "Unable to apply a dropout to layer 0")
	}

	if c1, err = nnops.Conv2d(l0, cnn.W1, tensor.Shape{3, 3},	// 第1层卷积
	[]int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 1 Convolution failed")
	}
	if a1, err = nnops.Rectify(c1); err != nil {	// 第1层激活
		return errors.Wrap(err, "Layer 1 activation failed")
	}
	if p1, err = nnops.MaxPool2D(a1, tensor.Shape{2, 2}, 	// 第1层最大池化
	[]int{0, 0}, []int{2, 2}); err != nil {
		return errors.Wrap(err, "Layer 1 Maxpooling failed")
	}
	if l1, err = nnops.Dropout(p1, cnn.D1); err != nil {		// 第1层概率退出
		return errors.Wrap(err, "Unable to apply a dropout to layer 1")
	}


	if c2, err = nnops.Conv2d(l1, cnn.W2, tensor.Shape{3, 3}, 	// 第2层卷积
	[]int{1, 1}, []int{1, 1}, []int{1, 1}); err != nil {
		return errors.Wrap(err, "Layer 2 Convolution failed")
	}
	if a2, err = nnops.Rectify(c2); err != nil {	// 第2层激活
		return errors.Wrap(err, "Layer 2 activation failed")
	}
	if p2, err = nnops.MaxPool2D(a2, tensor.Shape{2, 2},	// 第2层最大池化
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
	if l2, err = nnops.Dropout(r2, cnn.D2); err != nil {	// 第2层概率退出
		return errors.Wrap(err, "Unable to apply a dropout on layer 2")
	}
	_ = ioutil.WriteFile("output/tmp.dot", []byte(cnn.G.ToDot()), 0644)
	// 第3层
	if fc, err = gorgonia.Mul(l2, cnn.W3); err != nil {
		return errors.Wrapf(err, "Unable to multiply l2 and w3")
	}
	if a3, err = nnops.Rectify(fc); err != nil {	// 第3层激活
		return errors.Wrapf(err, "Unable to activate fc")
	}
	if l3, err = nnops.Dropout(a3, cnn.D3); err != nil {	// 第3层概率退出
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
	vm := gorgonia.NewTapeMachine(cnn.G, gorgonia.WithPrecompiled(prog, locMap),
		gorgonia.BindDualValues(cnn.Learnables()...))
	cnn.VM = vm
	return vm
}

func (cnn *CNN) Persistence() error {
	if err := cnnSave([]*gorgonia.Node{cnn.W0, cnn.W1, cnn.W2, cnn.W3, cnn.W4}, cnn.TrainEpoch); err != nil {
		return err
	}
	return nil
}

func LoadCNNFromSave() (*CNN, error) {
	f, err := os.Open("cnn_weights")
	if err != nil {
		return nil, err
	}
	defer f.Close()
	dec := gob.NewDecoder(f)
	var w0_val, w1_val, w2_val, w3_val, w4_val *tensor.Dense
	var trainEpoch int
	log.Println("decoding w0")
	if err = dec.Decode(&w0_val); err != nil {
		return nil, err
	}
	log.Println("decoding w1")
	if err = dec.Decode(&w1_val); err != nil {
		return nil, err
	}
	log.Println("decoding w2")
	if err = dec.Decode(&w2_val); err != nil {
		return nil, err
	}
	log.Println("decoding w3")
	if err = dec.Decode(&w3_val); err != nil {
		return nil, err
	}
	log.Println("decoding w4")
	if err = dec.Decode(&w4_val); err != nil {
		return nil, err
	}
	log.Println("decoding trainEpoch")
	if err = dec.Decode(&trainEpoch); err != nil {
		return nil, err
	}
	g := gorgonia.NewGraph()
	x := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithShape(common.CNNBatchSize,
		common.RawImageChannel, common.RawImageRows,
		common.RawImageCols), gorgonia.WithName("x"))
	// 表达式网络输入数据y，内容为上述图像数据对应标签张量
	y := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithShape(common.CNNBatchSize,
		common.EMNISTByClassNumLabels), gorgonia.WithName("y"))
	w0:= gorgonia.NewTensor(g, tensor.Float64,4, gorgonia.WithValue(w0_val), gorgonia.WithName("w0"))
	log.Printf(" w0:%v \n", w0)
	w1 := gorgonia.NewTensor(g, tensor.Float64,4, gorgonia.WithValue(w1_val), gorgonia.WithName("w1"))
	log.Printf("w1:%v \n", w1)
	w2 := gorgonia.NewTensor(g, tensor.Float64, 4, gorgonia.WithValue(w2_val), gorgonia.WithName("w2"))
	log.Printf("w2:%v \n", w2)
	w3 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithValue(w3_val), gorgonia.WithName("w3"))
	log.Printf("w3:%v \n", w3)
	w4 := gorgonia.NewMatrix(g, tensor.Float64, gorgonia.WithValue(w4_val), gorgonia.WithName("w4"))
	log.Printf("w4:%v \n", w4)
	cnn := &CNN{
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
		TrainEpoch: trainEpoch,
	}
	if err := cnn.Fwd(x); err != nil {
		return nil, err
	}
	losses := gorgonia.Must(gorgonia.HadamardProd(cnn.Out, y))
	cost := gorgonia.Must(gorgonia.Mean(losses))
	cost = gorgonia.Must(gorgonia.Neg(cost))
	// 跟踪成本
	var costVal gorgonia.Value
	gorgonia.Read(cost, &costVal)
	// 根据权重矩阵获取成本
	if _, err := gorgonia.Grad(cost, cnn.Learnables()...); err != nil {
		return nil, err
	}
	//var out gorgonia.Value
	//rv := gorgonia.Read(cnn.Out, &out)
	//fwdNet := cnn.G.SubgraphRoots(rv)
	//vm := gorgonia.NewTapeMachine(fwdNet)
	//cnn.VM = vm
	cnn.VMBuild()
	return cnn, nil
}

func cnnSave(nodes []*gorgonia.Node, trainEpoch int) error {
	f, err := os.Create("cnn_weights")
	if err != nil {
		return err
	}
	defer f.Close()
	enc := gob.NewEncoder(f)
	for _, node := range nodes {
		err := enc.Encode(node.Value())
		if err != nil {
			return err
		}
	}
	err = enc.Encode(trainEpoch)
	if err != nil {
		return err
	}
	return nil
}