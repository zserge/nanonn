package nn

import (
	"encoding/csv"
	"math"
	"math/rand"
	"os"
	"strconv"
	"testing"
	"time"
)

func init() { rand.Seed(time.Now().UnixNano()) }

func TestSigmoid(t *testing.T) {
	if y := sigmoid(0); y != 0.5 {
		t.Error(y)
	}
	if y := sigmoid(2); math.Abs(y-0.88079708) > 0.0001 {
		t.Error(y)
	}
}

func TestForward(t *testing.T) {
	l := Dense(1, 3, Sigmoid())
	l.SetWeights([]float64{1.74481176, -0.7612069, 0.3190391, -0.24937038})
	x1 := []float64{1.62434536, -0.52817175, 0.86540763}
	y1 := 0.96313579
	z1 := l.Forward(x1)
	if math.Abs(z1[0]-y1) > 0.001 {
		t.Error(z1, y1)
	}
	x2 := []float64{-0.61175641, -1.07296862, -2.3015387}
	y2 := 0.22542973
	z2 := l.Forward(x2)
	if math.Abs(z2[0]-y2) > 0.001 {
		t.Error(z2, y2)
	}
}

// https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
func TestWeights(t *testing.T) {
	l1 := Dense(2, 2)
	l2 := Dense(2, 2)
	n, _ := New(l1, l2)

	// Initialise weights
	l1.SetWeights([]float64{0.15, 0.2, 0.35, 0.25, 0.3, 0.35})
	l2.SetWeights([]float64{0.4, 0.45, 0.6, 0.5, 0.55, 0.6})

	// Ensure forward propagation works for both layers
	z := n.Predict([]float64{0.05, 0.1})
	if e := math.Abs(z[0] - 0.75136507); e > 0.0001 {
		t.Error(e, z)
	}
	if e := math.Abs(z[1] - 0.772928465); e > 0.0001 {
		t.Error(e, z)
	}

	// Ensure that squared error is calculated correctly (use rate=0 to avoid training)
	if e := n.Train([]float64{0.05, 0.1}, []float64{0.01, 0.99}, 0); math.Abs(e-0.298371109) > 0.0001 {
		t.Log(e)
	}

	// Backpropagtion with rate 0.5
	n.Train([]float64{0.05, 0.1}, []float64{0.01, 0.99}, 0.5)

	// Check weights
	for i, w := range []float64{0.35891648, 0.408666186, 0.530751, 0.511301270, 0.561370121, 0.619049} {
		if math.Abs(l2.Weights()[i]-w) > 0.001 {
			t.Error(i, w, l2.Weights())
		}
	}
	for i, w := range []float64{0.149780716, 0.19956143, 0.345614, 0.24975114, 0.29950229, 0.345023} {
		if math.Abs(l1.Weights()[i]-w) > 0.001 {
			t.Error(i, w, l1.Weights())
		}
	}

}

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
	for epoch := 0; epoch < 10000; epoch++ {
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
		}
		return math.Sin(x) / x
	}
	n, _ := New(Dense(5, 1), Dense(10, 5), Dense(1, 10))
	for i := 0; i < 1000; i++ {
		e := 0.0
		for j := 0; j < 100; j++ {
			x := rand.Float64()*10 - 5
			e = e + n.Train([]float64{x}, []float64{sinc(x)}, 0.5)/100
		}
		if e < 0.01 {
			return
		}
	}
	t.Error("failed to train")
}

// Train and test on Iris dataset
func TestIris(t *testing.T) {
	x, y := loadCSV("../testdata/iris.csv")
	n, _ := New(Dense(10, 4), Dense(3, 10))
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

	e := 0.0
	for epoch := 0; epoch < 10000; epoch++ {
		e = 0.0
		for i := 0; i < k; i++ {
			e = e + n.Train(x[i], y[i], 0.4)/float64(k)
		}
		if e < 0.01 {
			// Classify all data and print failures
			for i := 0; i < len(x); i++ {
				z := n.Predict(x[i])
				if maxind(z) != maxind(y[i]) {
					t.Log(x[i], y[i], z)
				}
			}
			return
		}
	}
	t.Error("failed to train", e)
}

func BenchmarkTrain(b *testing.B) {
	n, _ := New(Dense(10, 1), Dense(3, 10), Dense(1, 3))
	x := []float64{0}
	y := []float64{0}
	for i := 0; i < b.N; i++ {
		x[0] = rand.Float64()*10 - 5
		y[0] = math.Sin(x[0])
		n.Train(x, y, 0.5)
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
