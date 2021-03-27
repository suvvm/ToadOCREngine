package main

import (
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/method"
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
	nn.SNNTrainingTest(dataZCA,datalabs)
}
