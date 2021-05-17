package method

import (
	"context"
	"crypto/md5"
	"encoding/hex"
	"fmt"
	"strconv"
	"suvvm.work/toad_ocr_engine/dal/cluster"
	"suvvm.work/toad_ocr_engine/dal/db"
	"suvvm.work/toad_ocr_engine/model"
)

func VerifySecret(ctx context.Context, appID, basicToken, verifyStr string) error {
	idStr, err := strconv.Atoi(appID)
	if err != nil {
		return fmt.Errorf("appID not int %v", err)
	}
	appInfo := &model.AppInfo{}
	appInfo.ID = idStr
	appSecret, err := cluster.GetKV(ctx, strconv.Itoa(appInfo.ID))
	if err != nil {
		appInfo, err = db.GetAppInfo(appInfo)
		if err != nil {
			return fmt.Errorf("appID not exists %v", err)
		}
		cluster.PutKV(ctx, strconv.Itoa(appInfo.ID), appSecret)
		appSecret = appInfo.Secret
	}
	hasher := md5.New()
	hasher.Write([]byte(appSecret + verifyStr))
	md5Token := hex.EncodeToString(hasher.Sum(nil))
	fmt.Printf("md5Token:%v", md5Token)
	if  md5Token != basicToken {
		return fmt.Errorf("basic token incompatible")
	}
	return nil
}

