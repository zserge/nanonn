package nn

import (
	"encoding/csv"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
)

// Use single unit layer to predict OR function
func TestLayerOr(t *testing.T) {
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float64{{0}, {1}, {1}, {1}}
	e := make([]float64, 1, 1)
	// A layer of a single unit with two x and one output
	l := NewDense(1, 2)
	for iter := 0; iter < 1000; iter++ {
		for i, x := range x {
			z := l.Forward(x)
			e[0] = y[i][0] - z[0]
			l.Backward(x, e, 1)
		}
	}
	for i, x := range x {
		z := l.Forward(x)
		t.Log(z, y[i])
		if math.Abs(z[0]-y[i][0]) > 0.1 {
			t.Error(x, z, y[i])
		}
	}
}

func TestNetworkXor(t *testing.T) {
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float64{{0}, {1}, {1}, {0}}
	n := Network{NewDense(4, 2), NewDense(1, 4)}
	for iter := 0; iter < 1000; iter++ {
		for i := range x {
			n.Train(x[i], y[i], 1)
		}
	}
	for i := range x {
		z := n.Predict(x[i])
		t.Log(z, y[i])
		if math.Abs(z[0]-y[i][0]) > 0.1 {
			t.Error(x, z, y[i])
		}
	}
}

func TestNetworkSinc(t *testing.T) {
	sinc := func(x float64) float64 {
		if x == 0 {
			return 1
		} else {
			return math.Sin(x) / x
		}
	}
	n := Network{NewDense(5, 1), NewDense(10, 5), NewDense(1, 10)}
	eval := func() float64 {
		e := float64(0)
		for i := 0; i < 100; i++ {
			x := rand.Float64()*2 - 1
			y := sinc(x)
			z := n.Predict([]float64{x})
			e = e + math.Abs(z[0]-y)
		}
		return e / 100
	}
	for i := 0; i < 100; i++ {
		x := rand.Float64()*2 - 1
		n.Train([]float64{x}, []float64{sinc(x)}, 1)
		e := eval()
		if i%10 == 0 {
			t.Log(e)
		}
	}
}

func TestIris(t *testing.T) {
	x, y := loadCSV("../testdata/iris.csv")
	n := Network{NewDense(10, 4), NewDense(5, 10), NewDense(3, 5)}
	k := len(x) * 9 / 10
	for i := range y {
		n := y[i][0]
		y[i] = []float64{0, 0, 0}
		y[i][int(n)] = 1
		x[i] = x[i][1:]
	}
	rand.Shuffle(len(x), func(i, j int) {
		x[i], x[j] = x[j], x[i]
		y[i], y[j] = y[j], y[i]
	})
	maxind := func(x []float64) int {
		m := -1
		for i := range x {
			if m < 0 || x[i] > x[m] {
				m = i
			}
		}
		return m
	}
	eval := func() (e float64) {
		for i := 0; i < len(x); i++ {
			z := n.Predict(x[i])
			if maxind(z) != maxind(y[i]) {
				e++
			}
		}
		return e
	}
	for iter := 0; iter < 200000; iter++ {
		i := rand.Intn(k)
		n.Train(x[i], y[i], 0.1)
		if iter%1000 == 0 {
			t.Log(eval())
		}
	}
	if eval() > 5 {
		t.Fail()
	}
}

func loadCSV(filename string) (x [][]float64, y [][]float64) {
	f, _ := os.Open(filename)
	defer f.Close()
	rows, _ := csv.NewReader(f).ReadAll()
	for _, row := range rows {
		nums := []float64{}
		for _, s := range row {
			n, _ := strconv.ParseFloat(s, 64)
			nums = append(nums, n)
		}
		x = append(x, nums[0:len(nums)-1])
		y = append(y, nums[len(nums)-1:])
	}
	return x, y
}
