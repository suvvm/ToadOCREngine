package utils

import (
	"encoding/binary"
	"io"
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
//	[]MNISTLabel				// 读取的标签
//	error				// 错误信息
func ReadLabelFile(reader io.Reader) ([]common.MNISTLabel, error) {
	var magic, n int32
	// 在数据源reader中大端序读出文件幻数（长度为size of int32的数据）
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != common.MNISTLabelMagic { // 文件幻数与标签文件幻数不符，证明当前读取的文件非标签文件
		return nil, os.ErrInvalid
	}
	// 在数据源reader中大端序读出标签数量
	if err :=  binary.Read(reader, binary.BigEndian, &n); err != nil {
		return nil, err
	}
	labels := make([]common.MNISTLabel, n)
	// 读取标签至labels
	for i := 0; i < int(n); i++ {
		var l common.MNISTLabel
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
//	[]MNISTRawImage			// 读取的图像
//	error				// 错误信息
func ReadImageFile(reader io.Reader) ([]common.MNISTRawImage, error) {
	var magic, n, nrow, ncol int32
	// 在数据源reader中大端序读出文件幻数（长度为size of int32的数据）
	if err := binary.Read(reader, binary.BigEndian, &magic); err != nil {
		return nil, err
	}
	if magic != common.MNISTImageMagic { // 文件幻数与图像文件幻数不符，证明当前读取的文件非图像文件
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
	images := make([]common.MNISTRawImage, n)
	imageSize := int(nrow * ncol)	// 图像大小
	for i := 0; i < int(n); i++ {
		images[i] = make(common.MNISTRawImage, imageSize)
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
