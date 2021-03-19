package utils

import (
	"encoding/binary"
	"gorgonia.org/tensor"
	"image"
	"image/color"
	"image/png"
	"io"
	"os"
	"suvvm.work/toad_ocr_engine/model"
)

type RawImage []byte	// 原始图像
type Label uint8		// mnist 数字标签 范围0～9

const numLabels = 10	// 数字标签个数
const pixelRange = 255	// 像素强度范围
const rawImageRows = 28	// 原始图像高度28像素
const rawImageCols = 28	// 原始图像宽度28像素

const (
	imageMagic = 0x00000803	// 图像文件幻数
	labelMagic = 0x00000801	// 标签文件幻数
	Width = 28				// 图像高度
	Height = 28				// 图像宽度
)

// ReadLabelFile 读取标签文件
//
// 入参
//	reader io.Reader	// 可读出字节流的数据源
// 文件幻数、标签数量都是文件中的元数据
//
// 返回
//	[]Label				// 读取的标签
//	error				// 错误信息
func ReadLabelFile(reader io.Reader) ([]Label, error) {
	var magic, n int32
	// 在数据源reader中大端序读出文件幻数（长度为size of int32的数据）
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != labelMagic {	// 文件幻数与标签文件幻数不符，证明当前读取的文件非标签文件
		return nil, os.ErrInvalid
	}
	// 在数据源reader中大端序读出标签数量
	if err :=  binary.Read(reader, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels := make([]Label, n)
	// 读取标签至labels
	for i := 0; i < int(n); i++ {
		var l Label
		if err := binary.Read(reader, binary.BigEndian, &l); err != nil {
			return nil, err
		}
		labels[i] = l
	}
	return labels, nil
}

// ReadImageFile 读取图像文件
//
// 入参
//	reader io.Reader	// 可读出字节流的数据源
// 文件幻数、图像数量、图像长度、图像宽度都是文件中的元数据
//
// 返回
//	[]RawImage			// 读取的图像
//	error				// 错误信息
func ReadImageFile(reader io.Reader) ([]RawImage, error) {
	var magic, n, nrow, ncol int32
	// 在数据源reader中大端序读出文件幻数（长度为size of int32的数据）
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != imageMagic {	// 文件幻数与图像文件幻数不符，证明当前读取的文件非图像文件
		return nil, os.ErrInvalid
	}
	// 在数据源reader中大端序读出图像数量
	if err := binary.Read(reader, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	// 在数据源reader中大端序读出图像宽度
	if err := binary.Read(reader, binary.BigEndian, &nrow); err != nil {
		return nil, err
	}
	// 在数据源reader中大端序读出图像高度
	if err := binary.Read(reader, binary.BigEndian, &ncol); err != nil {
		return nil, err
	}
	images := make([]RawImage, n)
	imageSize := int(nrow * ncol)	// 图像大小
	for i := 0; i < int(n); i++ {
		images[i] = make(RawImage, imageSize)
		readSize, err := io.ReadFull(reader, images[i])	// 读取一个图像
		if err != nil {
			return nil, err
		}
		if readSize != imageSize {	// 读取长度与图像大小不一致
			return nil, os.ErrInvalid
		}
	}
	return images, nil
}


// PixelWeight 像素灰度缩放
// mnist给出的图像作为灰度图，单个像素点通过8位的灰度值(0~255)来表示。
// PixelWeight 函数将字节类型的像素灰度值转换为float64类型，并将范围缩放至(0.0~1.0)
//
// 入参
//	px byte	// 字节型像素灰度值
//
// 返回
//	float64 // 缩放后的浮点灰度值
func PixelWeight(px byte) float64 {
	pixelVal := (float64(px) / pixelRange * 0.999) + 0.001
	if pixelVal == 1.0 {	// 如果缩放后的值为1.0时，为了数学性能的表现稳定，将其记为0.999
		return 0.999
	}
	return pixelVal
}

// ReversePixelWeight 像素灰度反向放大函数
// 将缩放后的像素灰度浮点恢复为8位字节灰度值
//
// 入参
//	px float64	// 浮点灰度值
//
// 返回
//	byte		// 8位字节灰度值
func ReversePixelWeight(px float64) byte {
	return byte((px - 0.001) / 0.999 * pixelRange)
}

// PrepareX 将图像数组转化为tensor存储的矩阵
// tensor设计格式详见 https://github.com/gorgonia/tensor#design-of-dense
//
// 入参
//	images []RawImage	// 完成初始化的mnist图像数组
//
// 返回
//	tensor.Tensor	// tensor矩阵
func PrepareX(images []RawImage) tensor.Tensor {
	rows := len(images)	// 矩阵宽度
	cols := len(images[0])	// 矩阵高度
	// 创建矩阵的支撑平面切片，tensor会复用当前切片的空间
	supportSlice := make([]float64, 0, rows * cols)
	for i := 0; i < rows; i++ {	// 复制缩放后的像素切片进入矩阵平面切片
		for j := 0; j < len(images[i]); j++ {
			supportSlice = append(supportSlice, PixelWeight(images[i][j]))
		}
	}
	// 返回rows * cols的tensor矩阵
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(supportSlice))
}

// PrepareY 将标签数组转化为tensor存储的矩阵
//
// 入参
//	labels []Label	// 完成初始化的mnist标签数组
//
// 返回
//	tensor.Tensor	// tensor矩阵
func PrepareY(labels []Label) tensor.Tensor {
	rows := len(labels)	// 矩阵宽度
	cols := numLabels	// 矩阵高度
	supportSlice := make([]float64, 0, rows * cols)	// 创建矩阵的支撑平面切片
	for i := 0; i < rows; i++ {	// 复制缩放后的像素切片进入矩阵平面切片
		for j := 0; j < numLabels; j++ {
			if j == int(labels[i]) {
				supportSlice = append(supportSlice, 1)
			} else {
				supportSlice = append(supportSlice, 0)
			}
		}
	}
	// 返回rows * cols的tensor矩阵
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(supportSlice))
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
	if err = sliced.Reshape(rows, cols, rawImageRows, rawImageCols); err != nil {
		return err
	}
	imRows := rawImageRows * rows
	imCols := rawImageCols * cols
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
				x := j * rawImageRows + k % rawImageRows
				y := i * rawImageCols + k / rawImageCols
				c := color.Gray{ReversePixelWeight(px)}
				canvas.Set(x, y, c)
			}
		}
	}
	var file io.WriteCloser
	if file, err = os.Create("output/"+filename); err != nil {
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
