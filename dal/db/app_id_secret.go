package db

import (
	"fmt"
	"github.com/google/uuid"
	"suvvm.work/toad_ocr_engine/model"
)

// AddAppInfo 存储app id与app secret信息
//
// 入参
//	appInfo *model.AppInfo	// 目标appInfo
// 返回
//	*model.AppInfo	// 目标appInfo
//	error		// 错误信息
func AddAppInfo(appInfo *model.AppInfo) (*model.AppInfo, error) {
	if appInfo.PNum == "" && appInfo.Email == "" {	// 判断appInfo是否完整
		return nil, fmt.Errorf("appInfo:missing require parameters")
	}
	appInfo.Secret = uuid.New().String()
	DB.Create(appInfo)	// 执行插入操作
	return appInfo, nil

}

// GetAppInfo 查询AppInfo
//
// 入参
//	appInfo *model.AppInfo	// 目标AppInfo
// 返回
//	*model.AppInfo		// 目标AppInfo完整信息
//	error		// 错误信息
func GetAppInfo(appInfo *model.AppInfo) (*model.AppInfo, error){
	var selectResp []model.AppInfo
	if appInfo.ID != 0 {	// 根据ID查询
		DB.Table("app_infos").Where("id=?", appInfo.ID).Select(
			[]string{"id", "secret", "email", "p_num"}).Find(&selectResp)
		if len(selectResp) == 0 {
			return nil, fmt.Errorf("appInfo:query id=%d, resp no datas", appInfo.ID)
		}
	} else if appInfo.Email != "" { //	根据email查询
		DB.Table("app_infos").Where("email=?", appInfo.Email).Select(
			[]string{"id", "secret", "email", "p_num"}).Find(&selectResp)
		if len(selectResp) == 0 {
			return nil, fmt.Errorf("appInfo:query email=%s, resp no datas", appInfo.Email)
		}
	} else { //	根据pnum查询
		DB.Table("app_infos").Where("p_num=?", appInfo.PNum).Select(
			[]string{"id", "secret", "email", "p_num"}).Find(&selectResp)
		if len(selectResp) == 0 {
			return nil, fmt.Errorf("appInfo:query pnum=%s, resp no datas", appInfo.PNum)
		}
	}
	return &selectResp[0], nil
}

// DelAppInfo 删除AppInfo
//
// 入参
//	appInfo *model.AppInfo	// 目标AppInfo
// 返回
//	error		// 错误信息
func DelAppInfo(appInfo *model.AppInfo) error {
	if appInfo.ID != 0 {	// 根据ID删除
		if err := DB.Delete(appInfo).Error; err != nil {
			return err
		}
	} else if appInfo.Email != "" {	// 根据email删除
		if err := DB.Where("email=?", appInfo.Email).Delete(&model.AppInfo{}).Error; err != nil {
			return err
		}
	} else {	// 根据手机号删除
		if err := DB.Where("p_num=?", appInfo.PNum).Delete(&model.AppInfo{}).Error; err != nil {
			return err
		}
	}
	return nil
}
