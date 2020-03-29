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
	l := Dense(1, 2)
	// Train layer for several epochs
	for epoch := 0; epoch < 1000; epoch++ {
		for i, x := range x {
			z := l.Forward(x)
			e[0] = y[i][0] - z[0]
			l.Backward(x, e, 1)
		}
	}
	// Predict the outputs, expecting only a small error
	for i, x := range x {
		z := l.Forward(x)
		if math.Abs(z[0]-y[i][0]) > 0.1 {
			t.Error(x, z, y[i])
		}
	}
}

// Use hidden layer to predict XOR function
func TestNetworkXor(t *testing.T) {
	x := [][]float64{{0, 0}, {0, 1}, {1, 0}, {1, 1}}
	y := [][]float64{{0}, {1}, {1}, {0}}
	n, _ := New(Dense(4, 2), Dense(1, 4))
	// Train for several epochs, or until the error is less than 2%
	for epoch := 0; epoch < 1000; epoch++ {
		e := 0.0
		for i := range x {
			e = e + n.Train(x[i], y[i], 1)
		}
		if e < 0.02 {
			return
		}
	}
	t.Error("failed to train")
}

// Use multiple hidden layers to predict sinc(x) function.
func TestNetworkSinc(t *testing.T) {
	sinc := func(x float64) float64 {
		if x == 0 {
			return 1
		} else {
			return math.Sin(x) / x
		}
	}
	n, _ := New(Dense(5, 1), Dense(10, 5), Dense(1, 10))
	e := 0.0
	for i := 0; i < 2000; i++ {
		x := rand.Float64()*10 - 5
		e = e + n.Train([]float64{x}, []float64{sinc(x)}, 1)
		if i%10 == 9 {
			if e/10 < 0.001 {
				return
			}
			t.Log(e / 10)
			e = 0
		}
	}
	t.Error("failed to train")
}

// Train and test on Iris dataset
func TestIris(t *testing.T) {
	x, y := loadCSV("../testdata/iris.csv")
	n, _ := New(Dense(10, 4), Dense(5, 10), Dense(3, 5))
	k := len(x) * 9 / 10 // use 90% for training, 10% for testing
	// replace Y with a 3-item vector for classification
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

	for epoch := 0; epoch < 10000; epoch++ {
		e := 0.0
		for i := 0; i < k; i++ {
			e = e + n.Train(x[i], y[i], 0.1)
		}
		if e/float64(k) < 0.005 {
			// Classify all data
			for i := 0; i < len(x); i++ {
				z := n.Predict(x[i])
				if maxind(z) != maxind(y[i]) {
					t.Log(x[i], y[i], z)
				}
			}
			return
		}
		if epoch%10 == 0 {
			t.Log(e / float64(k))
		}
	}
	t.Error("failed to train")
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
