package model

import "gorgonia.org/gorgonia"

// CNN 四层卷积神经网络
// 卷积层表达式DropOut(MaxPoll(\sigma(w*x)))
// 退出与最大池化为同一层
type CNN struct {
	G	*gorgonia.ExprGraph				// 语法数学表达式图（可重用节点的AST）
	W0, W1, W2, W3, W4 *gorgonia.Node	// 各层权重
	D0, D1, D2, D3 float64				// 退出概率
	Out *gorgonia.Node
	OutVal gorgonia.Value
}
