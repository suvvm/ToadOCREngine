package main

import (
	"log"
	"os"
	"suvvm.work/toad_ocr_engine/common"
	"suvvm.work/toad_ocr_engine/model"
	"suvvm.work/toad_ocr_engine/nn"
	"suvvm.work/toad_ocr_engine/utils"
)

func main() {
	if len(os.Args) == 1 {
		log.Printf("Please provide command parameters\n Running with " +
			"`help` to show currently supported commands")
		return
	}
	cmd := os.Args[1]
	if _, ok := common.CMDMap[cmd]; !ok {
		log.Printf("Unknow command!\n")
		return
	}
	if cmd == common.CmdHelp {	// 帮助命令
		log.Printf("\nToad OCR Engine Help:\ntrain: use command `%s` + target neural networks" +
			" to training networks(use mnist train set when training)\n" +
			"test: use command `%s` + target neural networks to testing" +
			" networks(use mnist test set when testing)\n" +
			"reset: use command `%s` + target neural networks to delete the weights file\n" +
			"nnlist: use command `%s` to show all supported networks",
			common.CmdTrain, common.CmdTest, common.CmdReset, common.CmdList)
		return
	} else if cmd == common.CmdList {	// 展示支持的神经网络
		log.Printf("supported networks:\n")
		for key, _ := range common.NNMap {
			log.Printf("\t%s\n", key)
		}
		return
	}
	if len(os.Args) < 3 {	// 判断target neural networks
		log.Printf("Please provide target neural networks\n")
		return
	}
	cmdnn := os.Args[2]
	if _, ok :=  common.NNMap[cmdnn]; !ok {	// 检查神经网络种类
		log.Printf("Please confirm the name of neural networks, use cmd " +
			"`nnlist` to show all supported networks\n")
		return
	}
	if cmd == common.CmdTrain {	// 训练命令
		if cmdnn == common.CnnName {
			nn.RunCNN()
		} else if cmdnn == common.SnnName {
			nn.RunSNN()
		}
	} else if cmd == common.CmdTest {	// 测试命令
		_, _, testData, testLbl := utils.LoadMNIST(common.MNSITTrainImagesPath,
			common.MNISTTrainLabelsPath, common.MNISTTestImagesPath, common.MNISTTestLabelsPath)
		if cmdnn == common.SnnName {
			_, err 	:= os.Stat("snn_weights")
			if err != nil && !os.IsExist(err){
				log.Printf("Please training first!\n")
				return
			}
			snn, err := model.LoadSNNFromSave()
			if err != nil {
				log.Fatalf("Failed at load snn weights %v", err)
			}
			nn.SNNTesting(snn, testData, testLbl)
		} else if cmdnn == common.CnnName {
			_, err 	:= os.Stat("cnn_weights")
			if err != nil && !os.IsExist(err){
				log.Printf("Please training first!\n")
				return
			}
			cnn, err := model.LoadCNNFromSave()
			defer cnn.VM.Close()
			if err != nil {
				log.Fatalf("Unable to load cnn file %v", err)
			}
			nn.CNNTesting(cnn, testData, testLbl)
		}
	} else if cmd == common.CmdReset {	// 重置命令(清除权重矩阵)
		if cmdnn == common.CnnName {
			_, err 	:= os.Stat("cnn_weights")
			if err != nil && !os.IsExist(err){
				log.Printf("Cnn weights file had been deleted!\n")
				return
			}
			err = os.Remove("cnn_weights")
			if err != nil {
				log.Printf("Failed to delete cnn weights:%v\n", err)
			}
			log.Printf("Cnn weights file deleted!\n")
		} else if cmdnn == common.SnnName {
			_, err 	:= os.Stat("snn_weights")
			if err != nil && !os.IsExist(err){
				log.Printf("Snn weights file had been deleted!\n")
				return
			}
			err = os.Remove("snn_weights")
			if err != nil {
				log.Printf("Failed to delete snn weights:%v\n", err)
			}
			log.Printf("Snn weights file deleted!\n")
		}
	} else {
		log.Printf("Unknow command!\n")
	}
}
