function clone_many_times(net, T)
  local clones = {}

  local params, gradParams
  if net.parameters then
    params, gradParams = net:parameters()
    if params == nil then
      params = {}
    end
  end

  local paramsNoGrad
  if net.parametersNoGrad then
    paramsNoGrad = net:parametersNoGrad()
  end

  -- look for all masks (pruned linear)
  local masksWeight = {}
  local masksBias = {}
  net:apply(function(m)
    if (m.negmaskWeight) then table.insert(masksWeight, m.negmaskWeight) end
    if (m.negmaskBias) then table.insert(masksBias, m.negmaskBias) end
  end)

  local mem = torch.MemoryFile("w"):binary()
  mem:writeObject(net)

  for t = 1, T do
    -- We need to use a new reader for each clone.
    -- We don't want to use the pointers to already read objects.
    local reader = torch.MemoryFile(mem:storage(), "r"):binary()
    local clone = reader:readObject()
    reader:close()

    if net.parameters then
      local cloneParams, cloneGradParams = clone:parameters()
      local cloneParamsNoGrad
      for i = 1, #params do
        cloneParams[i]:set(params[i])
        cloneGradParams[i]:set(gradParams[i])
      end
      if paramsNoGrad then
        cloneParamsNoGrad = clone:parametersNoGrad()
        for i =1,#paramsNoGrad do
          cloneParamsNoGrad[i]:set(paramsNoGrad[i])
        end
      end
    end

    -- if pruned models, use single copy of the boolean masks
    if #masksWeight>0 then
      local idxw=1
      local idxb=1
      clone:apply(function(m)
        if (m.negmaskWeight) then m.negmaskWeight=masksWeight[idxw];idxw=idxw+1 end
        if (m.negmaskBias) then m.negmaskBias=masksBias[idxb];idxb=idxb+1 end
      end)
    end

    clones[t] = clone
    collectgarbage()
  end

  mem:close()
  return clones
end

function adagrad_step(x, dfdx, lr, state)
  if not state.var then
    state.var = torch.Tensor():typeAs(x):resizeAs(x):zero()
    state.std = torch.Tensor():typeAs(x):resizeAs(x)
  end

  state.var:addcmul(1, dfdx, dfdx)
  state.std:sqrt(state.var)
  x:addcdiv(-lr, dfdx, state.std:add(1e-10))
end

function adam_step(x, dfdx, lr, state)
  local beta1 = state.beta1 or 0.9
  local beta2 = state.beta2 or 0.999
  local eps = state.eps or 1e-8

  state.t = state.t or 0
  state.m = state.m or x.new(dfdx:size()):zero()
  state.v = state.v or x.new(dfdx:size()):zero()
  state.denom = state.denom or x.new(dfdx:size()):zero()

  state.t = state.t + 1
  state.m:mul(beta1):add(1-beta1, dfdx)
  state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)
  state.denom:copy(state.v):sqrt():add(eps)

  local bias1 = 1-beta1^state.t
  local bias2 = 1-beta2^state.t
  local stepSize = lr * math.sqrt(bias2)/bias1
  x:addcdiv(-stepSize, state.m, state.denom)

end

function adadelta_step(x, dfdx, lr, state)
  local rho = state.rho or 0.9
  local eps = state.eps or 1e-6
  state.var = state.var or x.new(dfdx:size()):zero()
  state.std = state.std or x.new(dfdx:size()):zero()
  state.delta = state.delta or x.new(dfdx:size()):zero()
  state.accDelta = state.accDelta or x.new(dfdx:size()):zero()
  state.var:mul(rho):addcmul(1-rho, dfdx, dfdx)
  state.std:copy(state.var):add(eps):sqrt()
  state.delta:copy(state.accDelta):add(eps):sqrt():cdiv(state.std):cmul(dfdx)
  x:add(-lr, state.delta)
  state.accDelta:mul(rho):addcmul(1-rho, state.delta, state.delta)
end
