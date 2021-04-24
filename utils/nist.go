package utils

import (
	"encoding/binary"
	"gorgonia.org/tensor"
	"io"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
)

// ReadLabelFile 读取标签文件
//
// 入参
//	reader io.Reader	// 可读出字节流的数据源
// 文件幻数、标签数量都是文件中的元数据
//
// 返回
//	[]Label		// 读取的标签
//	error				// 错误信息
func ReadLabelFile(reader io.Reader) ([]common.Label, error) {
	var magic, n int32
	// 在数据源reader中大端序读出文件幻数（长度为size of int32的数据）
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != common.LabelMagic { // 文件幻数与标签文件幻数不符，证明当前读取的文件非标签文件
		return nil, os.ErrInvalid
	}
	// 在数据源reader中大端序读出标签数量
	if err :=  binary.Read(reader, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels := make([]common.Label, n)
	// 读取标签至labels
	for i := 0; i < int(n); i++ {
		var l common.Label
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
//	[]RawImage		// 读取的图像
//	error				// 错误信息
func ReadImageFile(reader io.Reader) ([]common.RawImage, error) {
	var magic, n, nrow, ncol int32
	// 在数据源reader中大端序读出文件幻数（长度为size of int32的数据）
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != common.ImageMagic { // 文件幻数与图像文件幻数不符，证明当前读取的文件非图像文件
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
	images := make([]common.RawImage, n)
	imageSize := int(nrow * ncol)	// 图像大小
	for i := 0; i < int(n); i++ {
		images[i] = make(common.RawImage, imageSize)
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

// PrepareX 将图像数组转化为tensor存储的矩阵
// tensor设计格式详见 https://github.com/gorgonia/tensor#design-of-dense
//
// 入参
//	images []RawImage	// 完成初始化的mnist图像数组
//
// 返回
//	tensor.Tensor	// tensor矩阵
func PrepareX(images []common.RawImage) tensor.Tensor {
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
func PrepareY(labels []common.Label, labelsCnt int) tensor.Tensor {
	rows := len(labels)                             // 矩阵宽度
	cols := labelsCnt                   // 矩阵高度
	supportSlice := make([]float64, 0, rows * cols) // 创建矩阵的支撑平面切片
	for i := 0; i < rows; i++ {                     // 复制缩放后的像素切片进入矩阵平面切片
		for j := 0; j < labelsCnt; j++ {
			if j == int(labels[i]) {
				supportSlice = append(supportSlice, 0.999)
			} else {
				supportSlice = append(supportSlice, 0.001)
			}
		}
	}
	// 返回rows * cols的tensor矩阵
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(supportSlice))
}

// EMNISTPrepareY 将标签数组转化为tensor存储的矩阵
//
// 入参
//	labels []Label	// 完成初始化的mnist标签数组
//
// 返回
//	tensor.Tensor	// tensor矩阵
func EMNISTPrepareY(labels []common.Label) tensor.Tensor {
	rows := len(labels)                             // 矩阵宽度
	cols := common.EMNISTByClassNumLabels                          // 矩阵高度
	supportSlice := make([]float64, 0, rows * cols) // 创建矩阵的支撑平面切片
	for i := 0; i < rows; i++ {                     // 复制缩放后的像素切片进入矩阵平面切片
		for j := 0; j < common.EMNISTByClassNumLabels; j++ {
			if j == int(labels[i]) {
				supportSlice = append(supportSlice, 0.999)
			} else {
				supportSlice = append(supportSlice, 0.001)
			}
		}
	}
	// 返回rows * cols的tensor矩阵
	return tensor.New(tensor.WithShape(rows, cols), tensor.WithBacking(supportSlice))
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
	pixelVal := (float64(px) / common.PixelRange * 0.999) + 0.001
	if pixelVal == 1.0 {	// 如果缩放后的值为1.0时，为了数学性能的表现稳定，将其记为0.999
		return 0.999
	}
	return pixelVal
}

// LoadNIST 读取MNIST文件为张量
//
// 入参
//	trainImgPath string		// 训练集图像数据地址
//	trainLabelsPath string	// 训练集标签数据地址
//	testImgPath string		// 测试集图像数据地址
//	testLabelsPath string	// 测试集标签数据地址
//
// 返回
//	dataImgs tensor.Tensor	// 训练集图像
//	dataLabs tensor.Tensor	// 训练集标签
//	testData tensor.Tensor	// 测试集图像
//	testLabs tensor.Tensor	// 测试集标签
func LoadNIST(trainImgPath, trainLabelsPath, testImgPath, testLabelsPath string)  (dataImgs, dataLabs, testData, testLabs tensor.Tensor) {
	// 读取图像文件
	reader, err := os.Open(trainImgPath)
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	images, err := ReadImageFile(reader)
	// 读取标签文件
	reader, err = os.Open(trainLabelsPath)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	labels, err := ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	// 打印图像与标签数量
	log.Printf("number of imgs:%d, number of labs:%d", len(images), len(labels))
	// 将图像与标签转化为张量
	dataImgs = PrepareX(images)
	dataLabs = PrepareY(labels, common.EMNISTByClassNumLabels)
	reader, err = os.Open(testImgPath)
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testImages, err := ReadImageFile(reader)
	if err != nil {
		log.Fatal(err)
	}
	reader, err = os.Open(testLabelsPath)
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	testlabels, err := ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	testData = PrepareX(testImages)
	testLabs = PrepareY(testlabels, common.EMNISTByClassNumLabels)
	return
}
