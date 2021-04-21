package main

import (
	"suvvm.work/toad_ocr_engine/nn"
)

//func main() {
//	// 读取图像文件
//	reader, err := os.Open("resources/mnist/train-images-idx3-ubyte")
//	if err != nil {
//		log.Fatalf("err:%s", err)
//	}
//	images, err := utils.ReadImageFile(reader)
//	// 读取标签文件
//	reader, err = os.Open("resources/mnist/train-labels-idx1-ubyte")
//	if err != nil {
//		log.Fatalf("err:%v", err)
//	}
//	labels, err := utils.ReadLabelFile(reader)
//	if err != nil {
//		log.Fatalf("err:%v", err)
//	}
//	// 打印图像与标签数量
//	log.Printf("number of imgs:%d, number of labs:%d", len(images), len(labels))
//	// 将图像与标签转化为张量
//	dataImgs := utils.PrepareX(images)
//	datalabs := utils.PrepareY(labels)
//
//	// 对图像进行ZCA白化
//	dataZCA, err := utils.ZCA(dataImgs)
//	if err != nil {
//		log.Fatalf("err:%v", err)
//	}
//
//	//// 可视化一个由10 * 10个原始图像组成的图像
//	//if err = method.Visualize(dataImgs, common.MNISTUnitRows, common.MNISTUnitCols, "image.png"); err != nil {
//	//	log.Fatalf("visualize error:%v", err)
//	//}
//	//// 可视化上方图像的ZCA白化版本
//	//if err = method.Visualize(dataZCA, common.MNISTUnitRows, common.MNISTUnitCols, "imageZCA.png"); err != nil {
//	//	log.Fatalf("visualize error:%v", err)
//	//}
//
//	// 构建一个三层基础神经网络
//	// 输入层神经元common.MNISTRawImageRows * common.MNISTRawImageCols个
//	// 隐层神经元common.MNISTRawImageRows * common.MNISTRawImageCols个
//	// 输出层神经元10个
//	snn := nn.NewSNN(common.MNISTRawImageRows * common.MNISTRawImageCols,
//		100, 10)
//	//snn := nn.NewSNN(common.MNISTRawImageRows * common.MNISTRawImageCols,
//	//	common.MNISTRawImageRows * common.MNISTRawImageCols, 10)
//	// 训练SNN10次
//	nn.SNNTraining(snn, dataImgs, dataZCA, datalabs, 10)
//	// 运行MNIST测试集并交叉验证
//	log.Printf("Start Testing")
//	// 读取MNIST测试集合
//	reader, err = os.Open("resources/mnist/t10k-images-idx3-ubyte")
//	if err != nil {
//		log.Fatalf("err:%s", err)
//	}
//	testImgs, err := utils.ReadImageFile(reader)
//	// 读取MNIST标签集合
//	reader, err = os.Open("resources/mnist/t10k-labels-idx1-ubyte")
//	if err != nil {
//		log.Fatalf("err:%s", err)
//	}
//	testLabs, err := utils.ReadLabelFile(reader)
//	// 将图像与标签转化为张量
//	testData := utils.PrepareX(testImgs)
//	testLbl := utils.PrepareY(testLabs)
//	// 执行测试
//	nn.SNNTesting(snn, testData, testLbl)
//}

func main() {
	//if len(os.Args) == 0 {
	//	log.Printf("Please provide command parameters\n Running with `help` to show currently supported commands")
	//	return
	//}
	//cmd := os.Args[0]
	//if cmd == "help" {
	//	log.Printf("")
	//	return
	//}
	nn.RunSNN()
}
