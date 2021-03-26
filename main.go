package main

import (
	"gorgonia.org/tensor"
	"gorgonia.org/tensor/native"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/method"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/nn"
	"suvvm.work/toad_ocr_engine/utils"
)

func main() {
	// 读取图像文件
	reader, err := os.Open("resources/mnist/train-images-idx3-ubyte")
	if err != nil {
		log.Fatalf("err:%s", err)
	}
	images, err := utils.ReadImageFile(reader)
	// 读取标签文件
	reader, err = os.Open("resources/mnist/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	labels, err := utils.ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	// 打印图像与标签数量
	log.Printf("number of imgs:%d, number of labs:%d", len(images), len(labels))
	// 将图像与标签转化为张量
	dataImgs := method.PrepareX(images)
	datalabs := method.PrepareY(labels)
	// 可视化一个由10 * 10个原始图像组成的图像
	if err = method.Visualize(dataImgs, 10, 10, "image.png"); err != nil {
		log.Fatalf("visualize error:%v", err)
	}
	// 对图像进行ZCA白化
	dataZCA, err := utils.ZCA(dataImgs)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	// 可视化上方图像的ZCA白化版本
	if err = method.Visualize(dataZCA, 10, 10, "imageZCA.png"); err != nil {
		log.Fatalf("visualize error:%v", err)
	}
	// 获取张量底层float64数组切片
	nat, err := native.MatrixF64(dataZCA.(*tensor.Dense))
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	log.Printf("SNN Start Training")
	// 构建一个三层基础神经网络
	// 输入层神经元common.MNISTRawImageRows * common.MNISTRawImageCols个
	// 隐层神经元100个
	// 输出层神经元10个
	snn := nn.New(common.MNISTRawImageRows * common.MNISTRawImageCols, 100, 10)
	// 构造成本数组
	costs := make([]float64, 0, dataZCA.Shape()[0])
	// 训练5组样本
	for i := 0; i < 5; i++ {
		dataZCAShape := dataZCA.Shape()
		var image, label tensor.Tensor
		for j := 0; j < dataZCAShape[0]; j++ {
			if image, err = dataZCA.Slice(model.MakeRS(j, j + 1)); err != nil {
				log.Fatalf("Unable to slice image %d", j)
			}
			if label, err = datalabs.Slice(model.MakeRS(j, j + 1)); err != nil {
				log.Fatalf("Unable to slice label %d", j)
			}
			var cost float64
			if cost, err = snn.SNNTrain(image, label, 0.1); err != nil {
				log.Fatalf("Training error:%v", err)
			}
			costs = append(costs, cost)

		}
		log.Printf("%d\tcosts:%v", i, utils.Avg(costs))
		utils.ShuffleX(nat)
		costs = costs[:0]
	}
	log.Printf("End Training!")
}
