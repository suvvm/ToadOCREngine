package common

type MNISTRawImage []byte // 原始图像
type MNISTLabel uint8     // mnist 数字标签 范围0～9

const (
	MNISTNumLabels    = 10         // 数字标签个数
	MNISTPixelRange   = 255        // 像素强度范围
	MNISTRawImageRows = 28         // 原始图像高度28像素
	MNISTRawImageCols = 28         // 原始图像宽度28像素
	MNISTImageMagic   = 0x00000803 // 图像文件幻数
	MNISTLabelMagic   = 0x00000801 // 标签文件幻数
	MNISTUnitRows     = 10         // 单个训练单元中每行包含原始图像的数量
	MNISTUnitCols     = 10         // 单个训练单元中列数
	MNISTRawImageChannel = 1       // 原始图像颜色通道数量

	BarMaxWidth       = 100        // 进度条最大宽度

	CNNBatchSize      = 100		   // CNN每个训练批次包含的原始图像数量
	CNNEpoch          = 10        // CNN训练阶段数量
)
