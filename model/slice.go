package model

// RangedSlice 实现了tensor.Slice接口的范围切片
type RangedSlice struct {
	start, end, step int
}

func (s RangedSlice) Start() int { return s.start }
func (s RangedSlice) End() int   { return s.end }
func (s RangedSlice) Step() int  { return s.step }

// MakeRS 创建一个范围切片
//
// 入参
//	start int	// 切片起点
//	end int		// 切片终点
//	opts ...int	// 可选参数
//
// 返回
//	RangedSlice 范围切片
func MakeRS(start, end int, opts ...int) RangedSlice {
	step := 1
	if len(opts) > 0 {
		step = opts[0]
	}
	return RangedSlice{
		start: start,
		end:   end,
		step:  step,
	}
}
