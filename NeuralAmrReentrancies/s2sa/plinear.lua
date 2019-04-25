-- Pruned Linear - extend the Linear class by overriding reset/accGradParameters methods
-- it prevents gradient propagation through the cut connections
-- negmask(Weight|Bias) is a byte mask, where a value 1 means that the connection is cut - the
-- corresponding weight|bias is set to 0

function nn.Linear:prune(val)
  if self.negmaskWeight == nil then
    -- create the boolean mask using some GPU setting than the actual storage
    self.negmaskWeight = torch.ByteTensor(self.weight:size()):zero()
    if string.find(torch.typename(self.weight),"Cuda") ~= nil then
      local curDevice=cutorch.getDevice()
      cutorch.setDevice(self.weight:getDevice())
      self.negmaskWeight=self.negmaskWeight:cuda()
      cutorch.setDevice(curDevice)
    end
  end
  if self.bias and self.negmaskBias == nil then
    -- create the boolean mask using some GPU setting than the actual storage
    self.negmaskBias = torch.ByteTensor(self.bias:size(1)):zero()
    if string.find(torch.typename(self.bias),"Cuda") ~= nil then
      local curDevice=cutorch.getDevice()
      cutorch.setDevice(self.bias:getDevice())
      self.negmaskBias=self.negmaskBias:cuda()
      cutorch.setDevice(curDevice)
    end
  end
  for i=1,self.weight:size(1) do
    for j=1,self.weight:size(2) do
      if self.weight[i][j] <= val and self.weight[i][j] >= -val then
        self.negmaskWeight[i][j] = 1
      end
    end
    if self.bias and self.bias[i] <= val and self.bias[i] >= -val then
        self.negmaskBias[i] = 1
      end
  end
  self.weight:maskedFill(self.negmaskWeight, 0)
  if self.bias then
    self.bias:maskedFill(self.negmaskBias, 0)
  end
  return self:nPruned()
end

-- return the number of pruned parameters
function nn.Linear:nPruned()
  local c=0
  local t=0
  if self.negmaskWeight then
    c=c+self.negmaskWeight:sum()
    t=t+self.negmaskWeight:nElement()
  end
  if self.negmaskBias then
    c=c+self.negmaskBias:sum()
    t=t+self.negmaskBias:nElement()
  end
  return c, t
end

nn.Linear.reset_legacy=nn.Linear.reset
function nn.Linear:reset(stdv)
  self:reset_legacy(stdv)
  if self.negmaskWeight then
    self.negmaskWeight:fill(0)
  end
  if self.negmaskBias then
    self.negmaskBias:fill(0)
  end
  return self
end

nn.Linear.accGradParameters_legacy=nn.Linear.accGradParameters
function nn.Linear:accGradParameters(input, gradOutput, scale)
  self:accGradParameters_legacy(input, gradOutput, scale)
  -- reset any change on cut connections
  if self.negmaskWeight then
    self.weight:maskedFill(self.negmaskWeight, 0)
  end
  if self.negmaskBias then
    self.bias:maskedFill(self.negmaskBias, 0)
  end
end
