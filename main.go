package main

import (
	"fmt"
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/utils"
)

func main() {
	reader, err := os.Open("resources/mnist/train-images-idx3-ubyte")
	if err != nil {
		fmt.Errorf("err:%s", err)
		panic(err)
	}
	images, err := utils.ReadImageFile(reader)

	reader, err = os.Open("resources/mnist/train-labels-idx1-ubyte")
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	labels, err := utils.ReadLabelFile(reader)
	if err != nil {
		log.Fatalf("err:%v", err)
	}
	log.Printf("number of imgs:%d, number of labs:%d", len(images), len(labels))

	data := utils.PrepareX(images)
	if err = utils.Visualize(data, 10, 10, "image.png"); err != nil {
		log.Fatalf("visualize error:%v", err)
	}
}
