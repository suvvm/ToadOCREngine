package method

import (
	"gorgonia.org/tensor"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/model"
)

// ReversePixelWeight 像素灰度反向放大函数
// 将缩放后的像素灰度浮点恢复为8位字节灰度值
//
// 入参
//	px float64	// 浮点灰度值
//
// 返回
//	byte		// 8位字节灰度值
func ReversePixelWeight(px float64) byte {
	return byte((px - 0.001) / 0.999 * common.PixelRange)
}

// Visualize 图像可视化实现函数
// 给定行数和列数，在float64张量data中获取图像，对像素灰度反向放大，
// 最终拼接为一个包含rows * cols个原始图像，文件名为filename的png图像
//
// 入参
//	data tensor.Tensor	// 保存所有图像float64数据的张量
//	rows int			// 目标图像每行所包含原始图像的数量
//	cols int			// 目标图像每列所包含原始图像的数量
//	filename string		// 目标图像文件名
//
// 返回
// error				// 错误信息
func Visualize(data tensor.Tensor, rows, cols int, filename string) error {
	imageTotal := rows * cols	// 计算图像包含原始图像总数
	sliced := data
	var err error
	if imageTotal > 1 {
		sliced, err = data.Slice(model.MakeRS(0, imageTotal), nil)	// 对data张量切片
		if err != nil {
			return err
		}
	}
	// 将分割后的张量重新构造为一个四维切片，可以认为这是一个rows * cols的，
	// 由rawImageRows * rawImageCols的原始图像组成的行和列
	if err = sliced.Reshape(rows, cols, common.RawImageRows, common.RawImageCols); err != nil {
		return err
	}
	imRows := common.RawImageRows * rows
	imCols := common.RawImageCols * cols
	rect := image.Rect(0, 0, imCols, imRows)	// 构造对应大小的矩形图像
	canvas := image.NewGray(rect)	// 根据构造的矩形创建灰度图画布
	// 遍历图像才张量中提取像素灰度值进行填充
	for i := 0; i < cols; i++ {
		for j := 0; j < rows; j++ {
			var patch tensor.Tensor
			if patch, err = sliced.Slice(model.MakeRS(i, i + 1), model.MakeRS(j, j + 1)); err != nil {
				return err
			}
			patchData := patch.Data().([]float64)
			for k, px := range patchData {
				x := j * common.RawImageRows + k % common.RawImageRows
				y := i * common.RawImageCols + k / common.RawImageCols
				c := color.Gray{ReversePixelWeight(px)}
				canvas.Set(x, y, c)
			}
		}
	}
	// 构造png图像文件
	var file io.WriteCloser
	if file, err = os.Create("output/images/"+filename); err != nil {
		return err
	}
	if err = png.Encode(file, canvas); err != nil {
		return err
	}
	if err = file.Close(); err != nil {
		return err
	}
	return nil
}

// VisualizeWeights 保存神经网络训练结果(两个权重张量)并将其可视化
// 待完成
//
// 入参
//	w tensor.Tensor	// 权重张量
//	rows int		// 行数
//	cols int		// 列数
//	filename string	// 可视化图像文件名
//
// 返回
//	error			// 错误信息
func VisualizeWeights(w tensor.Tensor, rows, cols int, filename string) error {
	return nil
}
