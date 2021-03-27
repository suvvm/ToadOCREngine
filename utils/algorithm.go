package utils

import (
	"gonum.org/v1/gonum/stat/distuv"
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
	"gorgonia.org/vecf64"
	"math"
	"math/rand"
	"time"
)

// InverseSqrt 64位平方根倒数算法
//
// 入参
//	epsilon float64	// 规则化项，低通滤波处理
//
// 返回
//	func(float64) float64	// 完成epsilon处理的平方根倒数函数
func InverseSqrt(epsilon float64) func(float64) float64 {
	return func(a float64) float64 {
		return 1 / math.Sqrt(a + epsilon)
	}
}

// Avg 浮点数求平均值
//
// 入参
//	data []float64	// 需要求平均值的浮点数组
//
// 返回
//	float64			// 浮点数平均值
func Avg(data []float64) float64 {
	s := Sum(data)
	return s / float64(len(data))
}

// Sum 浮点数求和
//
// 入参
//	data []float64	// 需要求和的浮点数组
//
// 返回
//	float64			// 求和结果
func Sum(data []float64) float64 {
	var sum float64
	for i := range data {
		sum += data[i]
	}
	return sum
}

// Centralization 中心化函数
// 通过每个维度上的所有属性减去其平均值，实现将矩阵中心移到坐标原点
//
// 入参
//	data tensor.Tensor	// 需要中心化的目标张量
//
// 返回
//	error				// 错误信息
func Centralization(data tensor.Tensor) error {
	// 重新取得数据张量中的float64平面切片并将其转换为二维数组作为目标矩阵
	nat, err := native.MatrixF64(data.(*tensor.Dense))
	if err != nil {
		return err
	}
	for _, row := range nat {	// 遍历矩阵每一行
		center := Avg(row)
		vecf64.Trans(row, -center)	// 当前行所有值减去平均值
	}
	rows, cols := data.Shape()[0], data.Shape()[1]	// 获取data张量中的行数和列数
	center := make([]float64, cols)	// 每一列的中心点
	for i := 0; i < cols; i++ {	// 计算每一列的中心点
		var colCenter float64
		for j := 0; j < rows; j++ {
			colCenter += nat[j][i]
		}
		colCenter /= float64(rows)
		center[i] = colCenter
	}
	for _, row := range nat {
		vecf64.Sub(row, center)
	}
	return nil
}

// ZCA 零相位成分分析(zca)函数
// zca函数作用类似与主成分分析(pca)
// 首先进行pca主成分分析，将图像像素点张量作为特征进行数据降维
// 之后在将pca降维后的数据进行旋转，使得最终数据维度与原始数据维度相同
// 具体理论见本人博客 https://www.suvvm.work/2021/03/20/PCA%E4%B8%BB%E6%88%90%E5%88%86%E5%88%86%E6%9E%90/
//
// 入参
//	data tensor.Tensor	// 图像像素点张量
//
// 返回
//	tensor.Tensor		// ZCA白化后的张量
//	error				// 错误信息
func ZCA(data tensor.Tensor) (tensor.Tensor, error) {
	var dataTranspose, dataClone, sigma tensor.Tensor	// data转置 data备份 矩阵Σ(定义详见上述链接)
	dataClone = data.Clone().(tensor.Tensor)
	if err := Centralization(dataClone); err != nil {	// 对备份数据张量进行中心化操作
		return nil, err
	}
	var err error
	if dataTranspose, err =  tensor.T(dataClone); err != nil {	// 求得中心化完成后的矩阵的转置
		return nil, err
	}
	if sigma, err = tensor.MatMul(dataTranspose, dataClone); err != nil {	// 求矩阵Σ
		return nil, err
	}
	cols := sigma.Shape()[1]	// 获取矩阵Σ列数
	// 对矩阵Σ逐元素处以列数-1，并覆盖原矩阵Σ，最终得到协方差矩阵
	if _, err := tensor.Div(sigma, float64(cols-1), tensor.UseUnsafe()); err != nil {
		return nil, err
	}
	// S特征值矩阵 U特征向量矩阵 V=U‘ 奇异值分解后的V这里用不到
	S, U, _, err := sigma.(*tensor.Dense).SVD(true,true)	// 对协方差矩阵执行奇异值分解
	if err != nil {
		return nil, err
	}
	var diag, UTranspose, tmp tensor.Tensor
	// 规则化项epsilon取0.1 求平方根倒数得对角阵diag
	if diag, err = S.Apply(InverseSqrt(0.1), tensor.UseUnsafe()); err != nil {
		return nil, err
	}
	diag = tensor.New(tensor.AsDenseDiag(diag))
	if UTranspose, err = tensor.T(U); err != nil {	// 计算U的转置
		return nil, err
	}
	if tmp, err = tensor.MatMul(U, diag); err != nil {
		return nil, err
	}
	if tmp, err = tensor.MatMul(tmp, UTranspose); err != nil {
		return nil, err
	}
	if err = tmp.T(); err != nil {
		return nil, err
	}
	return tensor.MatMul(data, tmp)
}

// FillRandom 随机填充[]float64
// 使用连续均匀分布中的随机值填充[]float64
//
// 入参
//	data []float64	// 要填充的目标数组
//	val float64		// 随机值分布限制
func FillRandom(data []float64, val float64) {
	dist := distuv.Uniform{
		Min: -1 / math.Sqrt(val),
		Max: 1 / math.Sqrt(val),
	}
	for i := range data {	// 在分布范围内抽取随机数填充目标数组
		data[i] = dist.Rand()
	}
}

// Sigmoid （Logistic）神经网络激活函数
// 用于隐层神经元输出，将给定float64实数映射到(0,1)
// 定义公式 $ S(x) = \frac{1}{1 + e^{-x}} $
//
// 入参
//	x float64	// 给定float64实数
//
// 返回
//	float64		// 映射结果
func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-1 * x))
}

// DSigmoid Sigmoid的导数
//
// 入参
//	x float64	// 给定float64实数
//
// 返回
//	float64		// 运算结果
func DSigmoid(x float64) float64 {
	return (1 - x) * x
}

// Argmax 计算给定数据切片中最大值的索引
//
// 入参
//	data []float64	// 给定数据切片
//
// 返回
//	int				// 最大值的索引
// Deprecated: 不如用vecf64.Argmax
func Argmax(data []float64) int {
	var index int
	var max = math.Inf(-1)
	for i := range data {
		if data[i] > max {
			index = i
			max = data[i]
		}
	}
	return index
}

//64位平方根倒数速算法1.卡马克反转。基础是牛顿迭代法。
func sqrtRootFloat64(number float64) float64 {
	var i uint64
	var x, y float64
	f := 1.5
	x = number * 0.5
	y = number
	i = math.Float64bits(y)           //内存不变，浮点型转换成整型
	i = 0x5fe6ec85e7de30da - (i >> 1) //0x5f3759df,注意这一行，另一个数字是0x5f375a86
	y = math.Float64frombits(i)       //内存不变，浮点型转换成整型
	y = y * (f - (x * y * y))
	y = y * (f - (x * y * y))
	return number * y
}

// ShuffleX 随机排列函数
//
// 入参
//	data [][]float64	// 要随机排列的float64切片
func ShuffleX(data [][]float64) {
	r := rand.New(rand.NewSource(time.Now().UnixNano()))
	tmp := make([]float64, len(data[0]))
	for i := range data {
		j := r.Intn(i + 1)
		copy(tmp, data[i])
		copy(data[i], data[j])
		copy(data[j], tmp)
	}
}

//64位平方根倒数速算法2
func InvSqrt64(x1 float64) float64 {
	x := x1
	xhalf := 0.5 * x
	i := math.Float64bits(xhalf)      // get bits for floating VALUE
	i = 0x5fe6ec85e7de30da - (i >> 1) // gives initial guess y0
	x = math.Float64frombits(i)       // convert bits BACK to float
	x = x * (1.5 - xhalf*x*x)         // Newton step, repeating increases accuracy
	x = x * (1.5 - xhalf*x*x)         // Newton step, repeating increases accuracy
	x = x * (1.5 - xhalf*x*x)         // Newton step, repeating increases accuracy
	return 1 / x
}
