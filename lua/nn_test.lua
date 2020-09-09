local nn = require('nn')

local function assertEq(a, b, tol)
    tol = tol or 0
    assert(math.abs(a - b) <= tol,
        string.format('Failed: %f != %f.', a, b))
end

local function testDenseLayer()
    assertEq(4, 2 * 2)
end

local function testSigmoid()
    local actFn,y = nn.Sigmoid.act
    y = actFn(0)
    assertEq(0.5, y)

    y = actFn(2.0)
    assertEq(0.88079708, y, 0.0001)
end

local function testForward()
    local l = nn.Dense(3, 1, nn.Sigmoid)
    l.weights = { 1.74481176, -0.7612069, 0.3190391, -0.24937038 }

    local x1 = { 1.62434536, -0.52817175, 0.86540763 }
    local y1 = 0.96313579
    local z1 = l:forward(x1)
    assertEq(y1, z1[1], 0.001)

    local x2 = { -0.61175641, -1.07296862, -2.3015387 }
    local y2 = 0.22542973
    local z2 = l:forward(x2)
    assertEq(y2, z2[1], 0.001)
end

-- https://mattmazur.com/2015/03/17/a-step-by-step-backpropagation-example/
local function testWeights()
    local l1 = nn.Dense(2, 2)
    local l2 = nn.Dense(2, 2)
    local n = nn.Network(l1, l2)

    -- Initialize weights
    l1.weights = { 0.15, 0.2, 0.35, 0.25, 0.3, 0.35 }
    l2.weights = { 0.4, 0.45, 0.6, 0.5, 0.55, 0.6 }

    -- Ensure forward propagation works for both layers.
    local z = n:predict { 0.05, 0.1 }
    assertEq(0.75136507, z[1], 0.0001)
    assertEq(0.772928465, z[2], 0.0001)

    -- Ensure that squared error is calculated correctly (use rate=0 to avoid training).
    local e = n:train({ 0.05, 0.1 }, { 0.01, 0.99 }, 0)
    assertEq(0.298371109, e, 0.0001)

    -- Backpropagation with rate 0.5.
    n:train({ 0.05, 0.1 }, { 0.01, 0.99 }, 0.5)

    -- Check weights.
    for i,w in ipairs {
        0.35891648,
        0.408666186,
        0.530751,
        0.511301270,
        0.561370121,
        0.619049
    } do
        assertEq(w, l2.weights[i], 0.001)
    end

    for i,w in ipairs {
        0.149780716,
        0.19956143,
        0.345614,
        0.24975114,
        0.29950229,
        0.345023
    } do
        assertEq(w, l1.weights[i], 0.001)
    end
end

-- Use hidden layer to predict XOR function.
local function testNetworkXOR()
    local x = {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    }
    local y = {
        { 0 },
        { 1 },
        { 1 },
        { 0 }
    }
    local n = nn.Network {
        nn.Dense(2, 4),
        nn.Dense(4, 1)
    }

    -- Train for several epochs, or until the error is less than 2%.
    for i=1,10000 do
        local e = 0
        for i=1,#x do
            e = e + n:train(x[i], y[i], 1)
        end
        if e < 0.02 then
            return
        end
    end

    assert(false, "Failed to train the model!")
end

-- Run tests.
testDenseLayer()
testSigmoid()
testForward()
testWeights()
testNetworkXOR()