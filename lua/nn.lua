local math = math or require('math')

-- vec(n: number, [v: number|function(): number])
local function vec(n, v)
    local t = {}
    if type(v) == 'function' then
        for i=1,n do t[i] = v() end
    else
        v = v or 0
        for i=1,n do t[i] = v end
    end
    return t
end

-- vec2(n: number, m: number, [v: number|function(): number])
local function vec2(n, m, v)
    local t = {}
    for i=1,n do t[i] = vec(m, v) end
    return t
end

local nn = {
    -- Activation function tables.
    Linear = {
        act = function(x) return x end,
        dact = function(x) return 1 end
    },
    ReLU = {
        act = function(x) return x * (x > 0 and 1 or 0) end,
        dact = function(x) return 1 * (x > 0 and 1 or 0) end
    },
    LeakyReLU = {
        act = function(x) return x > 0 and x or (0.01 * x) end,
        dact = function(x) return x > 0 and 1 or 0.01 end
    },
    Sigmoid = {
        act = function(x) return 1 / (1 + math.exp(-x)) end,
        dact = function(x) return x * (1 - x) end
    },
    SoftPlus = {
        act = function(x) return math.log(1 + math.exp(x)) end,
        dact = function(x) return 1 / (1 + math.exp(-x)) end
    },
}

--- Network is a sequence of layers.
-- @param ... vector of layers or sequence of layers
-- @return Network
-- Ex:  nn.Network(l1, l2, l3)
--      nn.Network { l1, l2, l3 }
function nn.Network(...)
    local t = {...}
    local layers = (type(t[1]) == 'table' and t[1].__t == 'layer') and t or t[1]
    local n = #layers
    return {
        layers = layers or {},
        errors = vec(layers[n].outputsCount),
        inputs = vec2(n, n),

        --- Passes input vector into the network.
        -- @param ... input vector or sequence of numbers
        -- @return output vector
        -- Ex:  Network:predict(1, 2, 3)
        --      Network:predict { 1, 2, 3 }
        predict = function(self, ...)
            local x = {...}
            x = type(x[1]) == 'table' and x[1] or x
            for _,l in ipairs(self.layers) do
                x = l:forward(x)
            end
            return x
        end,

        --- Runs one backpropagation iteration through the network.
        -- @param x input vector
        -- @param y expected output vector
        -- @param rate learning rate
        -- @return training error
        train = function(self, x, y, rate)
            -- Localize properties
            local errors = self.errors
            local layers = self.layers
            local inputs = self.inputs
            rate = rate or 0
            for i,l in ipairs(layers) do
                inputs[i] = x
                x = l:forward(x)
            end
            local e = 0
            for i=1,#y do
                errors[i] = y[i] - x[i]
                e = e + errors[i] * errors[i]
            end
            for i=#layers,1,-1 do
                errors = layers[i]:backward(inputs[i], errors, rate)
            end
            return e / #y
        end
    }
end

-- Default implementation of a dense fully-connected layer
-- with sigmoid activation function
-- and the given number of inputs and neurons.
-- Ex:  nn.Dense(2, 2)
--      nn.Dense(2, 2, nn.SoftPlus)
function nn.Dense(inputsCount, outputsCount, actFn)
    local n = outputsCount * (inputsCount + 1)
    return {
        __t = 'layer',
        inputsCount = inputsCount,
        outputsCount = outputsCount,
        actFn = actFn or nn.Sigmoid,
        outputs = vec(outputsCount),
        errors = vec(inputsCount),
        -- Initialize weights randomly.
        weights = vec(n, function() return math.random() * 2 - 1 end),
        forward = function(self, x)
            local ic = self.inputsCount
            local oc = self.outputsCount
            local weights = self.weights
            local actFn = self.actFn.act
            local outputs = self.outputs
            local n = ic + 1
            for i=1,oc do
                local sum = 0
                for j=1,ic do
                    -- Lua index starts at 1.
                    -- [i * n + j] => [((i-1) * n + (j-1)) + 1]
                    sum = sum + x[j] * weights[(i-1) * n + j]
                end
                -- [i * n + n - 1] => [((i-1) * n + n - 1) + 1]
                outputs[i] = actFn(sum + weights[(i-1) * n + n])
            end
            return outputs
        end,
        backward = function(self, x, e, rate)
            local ic = self.inputsCount
            local oc = self.outputsCount
            local weights = self.weights
            local errors = self.errors
            local actFn = self.actFn.dact
            local outputs = self.outputs
            local n = ic + 1
            for j=1,ic do
                errors[j] = 0
                for i=1,oc do
                    errors[j] = errors[j] + e[i] * actFn(outputs[i]) * weights[(i-1) * n + j]
                end
            end
            for i=1,oc do
                for j=1,ic do
                    local idx = (i-1) * n + j
                    weights[idx] = weights[idx] + rate * e[i] * actFn(outputs[i]) * x[j]
                end
                local idx = (i-1) * n + n
                weights[idx] = weights[idx] + rate * e[i] * actFn(outputs[i])
            end
            return errors
        end
    }
end

return nn