package model

type AppInfo struct {
	ID int	`gorm:"column:id" json:"id"`
	Secret string `gorm:"column:secret" json:"secret"`
	Email string `gorm:"column:email" json:"email"`
	PNum string `gorm:"column:p_num" json:"p_num"`
}
