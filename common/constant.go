package common

type RawImage []byte          // 原始图像
type Label uint8              // mnist 数字标签 范围0～9

const (
	PixelRange      = 255        // 像素强度范围
	RawImageRows    = 28         // 原始图像高度28像素
	RawImageCols    = 28         // 原始图像宽度28像素
	ImageMagic      = 0x00000803 // 图像文件幻数
	LabelMagic      = 0x00000801 // 标签文件幻数
	RawImageChannel = 1          // 原始图像颜色通道数量

	MNISTNumLabels  = 10         // MNIST数字标签个数
	MNSITTrainImagesPath = "resources/mnist/train-images-idx3-ubyte"	// mnist训练图片数据路径
	MNISTTrainLabelsPath = "resources/mnist/train-labels-idx1-ubyte"	// mnist训练标签数据路径
	MNISTTestImagesPath  = "resources/mnist/t10k-images-idx3-ubyte"		// mnist测试图片数据路径
	MNISTTestLabelsPath  = "resources/mnist/t10k-labels-idx1-ubyte"		// mnist测试标签数据路径
	MNISTUnitRows   = 10         // 单个训练单元中每行包含原始图像的数量
	MNISTUnitCols   = 10         // 单个训练单元中列数

	EMNISTByClassNumLabels = 62         // EMNIST数字加字母标签总数
	EMNSITByClassTrainImagesPath = "resources/emnist/emnist-byclass-train-images-idx3-ubyte"	// mnist训练图片数据路径
	EMNISTByClassTrainLabelsPath = "resources/emnist/emnist-byclass-train-labels-idx1-ubyte"	// mnist训练标签数据路径
	EMNISTByClassTestImagesPath  = "resources/emnist/emnist-byclass-test-images-idx3-ubyte"		// mnist测试图片数据路径
	EMNISTByClassTestLabelsPath  = "resources/emnist/emnist-byclass-test-labels-idx1-ubyte"		// mnist测试标签数据路径

	BarMaxWidth       = 100        // 进度条最大宽度

	CNNBatchSize      = 100		   // CNN每个训练批次包含的原始图像数量
	CNNEpoch          = 100        // CNN训练阶段数量

	CmdTrain          = "train"
	CmdTest           = "test"
	CmdReset          = "reset"
	CmdList           = "nnlist"
	CmdHelp           = "help"

	CnnName           = "cnn"
	SnnName           = "snn"
)

var CMDMap = map[string]bool {
	CmdTrain: true,
	CmdTest: true,
	CmdReset: true,
	CmdList: true,
	CmdHelp: true,
}

var NNMap = map[string]bool {
	CnnName: true,
	SnnName: true,
}
