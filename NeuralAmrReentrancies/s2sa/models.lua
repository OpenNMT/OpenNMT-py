require 's2sa.memory'
require 's2sa.plinear'
-- local debugger = require("fb.debugger")
--kept for compatibilty reason with earlier model
--torch.class('nn.LinearNoBias', 'nn.Linear')

function make_encoder(data, opt)
  local x = nn.Identity()()   -- 57 x 100

  local word_vecs = nn.Bottle(nn.LookupTable(data.source_size, opt.word_vec_size), 1, 2)
  local lstm_out = cudnn.BLSTM(opt.word_vec_size, opt.rnn_size / 2, opt.num_layers)(nn.Dropout(opt.enc_dropout)(word_vecs(x)))
  
  return nn.gModule({x}, {lstm_out})
end

function make_lstm(data, opt, model, use_chars)
  assert(model == 'enc' or model == 'dec')
  local name = '_' .. model
  local dropout = opt.dropout or 0
  local n = opt.num_layers
  local rnn_size = opt.rnn_size
  local RnnD={rnn_size,rnn_size}
  local input_size
  if use_chars == 0 then
    input_size = opt.word_vec_size
  else
    input_size = opt.num_kernels
  end
  local offset = 0
  -- there will be 2*n+3 inputs
  local inputs = {}
  table.insert(inputs, nn.Identity()()) -- x (batch_size x max_word_l)
  if model == 'dec' then
    table.insert(inputs, nn.Identity()()) -- all context (batch_size x source_l x rnn_size)
    offset = offset + 1
    if opt.input_feed == 1 then
      table.insert(inputs, nn.Identity()()) -- prev context_attn (batch_size x rnn_size)
      offset = offset + 1
    end
  else
    for i = 1, data.num_source_features do
      table.insert(inputs, nn.Identity()()) -- table of features
      offset = offset + 1
    end
  end
  for L = 1,n do
    table.insert(inputs, nn.Identity()()) -- prev_c[L]
    table.insert(inputs, nn.Identity()()) -- prev_h[L]
  end

  local x, input_size_L, emb
  local outputs = {}
  for L = 1,n do
    local nameL=model..'_L'..L..'_'
    -- c,h from previous timesteps
    local prev_c = inputs[L*2+offset]
    local prev_h = inputs[L*2+1+offset]
    -- the input to this layer
    if L == 1 then
      if use_chars == 0 then
        local word_vecs
        if model == 'enc' then
          word_vecs = nn.LookupTable(data.source_size, input_size)
        else
          word_vecs = nn.LookupTable(data.target_size, input_size)
        end
        word_vecs.name = 'word_vecs' .. name
        x = word_vecs(inputs[1]) -- batch_size x word_vec_size
        emb = x
      else
        local char_vecs = nn.LookupTable(data.char_size, opt.char_vec_size)
        char_vecs.name = 'word_vecs' .. name
        local charcnn = make_cnn(opt.char_vec_size, opt.kernel_width, opt.num_kernels)
        charcnn.name = 'charcnn' .. name
        x = charcnn(char_vecs(inputs[1]))
        if opt.num_highway_layers > 0 then
          local mlp = make_highway(input_size, opt.num_highway_layers)
          mlp.name = 'mlp' .. name
          x = mlp(x)
        end
      end
      if model == 'enc' then
        for i = 1, data.num_source_features do
          local feat_vecs = nn.LookupTable(data.source_features_size[i],
                                           data.source_features_vec_size[i])
          local feat_x = feat_vecs(inputs[1+i])
          x = nn.JoinTable(2)({x, feat_x})
        end
      end
      input_size_L = input_size
      if model == 'dec' then
        if opt.input_feed == 1 then
          x = nn.JoinTable(2):usePrealloc("dec_inputfeed_join",
                                          {{opt.max_batch_l, opt.word_vec_size},{opt.max_batch_l, opt.rnn_size}})
                             ({x, inputs[1+offset]}) -- batch_size x (word_vec_size + rnn_size)
          input_size_L = input_size_L + rnn_size
        end
      else
        input_size_L = input_size_L + data.total_source_features_size
      end
    else
			x = outputs[(L-1)*2]
--       if opt.res_net == 1 and L > 2 then
--         x = nn.CAddTable()({x, outputs[(L-2)*2]})
--       end
      input_size_L = rnn_size
--       if opt.multi_attn == L and model == 'dec' then
--         local multi_attn = make_decoder_attn(data, opt, 1)
--         multi_attn.name = 'multi_attn' .. L
--         x = multi_attn({x, inputs[2]})
--       end
--       if dropout > 0 then
--         x = nn.Dropout(dropout, nil, false):usePrealloc(nameL.."dropout",
--                                                         {{opt.max_batch_l, input_size_L}})
--                                            (x)
--       end
    end
		if (opt.bow ~= 1 and model == 'enc') or model ==  'dec' then
			-- evaluate the input sums at once for efficiency
			local i2h = nn.Linear(input_size_L, 4 * rnn_size):usePrealloc(nameL.."i2h-reuse",
																																		{{opt.max_batch_l, input_size_L}},
																																		{{opt.max_batch_l, 4 * rnn_size}})
																											 (x)
			local h2h = nn.Linear(rnn_size, 4 * rnn_size, false):usePrealloc(nameL.."h2h-reuse",
																																			 {{opt.max_batch_l, rnn_size}},
																																			 {{opt.max_batch_l, 4 * rnn_size}})
																													(prev_h)
			local all_input_sums = nn.CAddTable():usePrealloc(nameL.."allinput",
																												{{opt.max_batch_l, 4*rnn_size},{opt.max_batch_l, 4*rnn_size}},
																												{{opt.max_batch_l, 4 * rnn_size}})
																					 ({i2h, h2h})

			local reshaped = nn.Reshape(4, rnn_size)(all_input_sums)
			local n1, n2, n3, n4 = nn.SplitTable(2):usePrealloc(nameL.."reshapesplit",
																													{{opt.max_batch_l, 4, rnn_size}})
																						 (reshaped):split(4)
			-- decode the gates
			local in_gate = nn.Sigmoid():usePrealloc(nameL.."G1-reuse",{RnnD})(n1)
			local forget_gate = nn.Sigmoid():usePrealloc(nameL.."G2-reuse",{RnnD})(n2)
			local out_gate = nn.Sigmoid():usePrealloc(nameL.."G3-reuse",{RnnD})(n3)
			-- decode the write inputs
			local in_transform = nn.Tanh():usePrealloc(nameL.."G4-reuse",{RnnD})(n4)
			-- perform the LSTM update
			local next_c = nn.CAddTable():usePrealloc(nameL.."G5a",{RnnD,RnnD})({
					nn.CMulTable():usePrealloc(nameL.."G5b",{RnnD,RnnD})({forget_gate, prev_c}),
					nn.CMulTable():usePrealloc(nameL.."G5c",{RnnD,RnnD})({in_gate, in_transform})
				})
			-- gated cells form the output
			local next_h = nn.CMulTable():usePrealloc(nameL.."G5d",{RnnD,RnnD})
																	 ({out_gate, nn.Tanh():usePrealloc(nameL.."G6-reuse",{RnnD})(next_c)})

			if opt.res_net == 1  then
				if L == 1 then
			 	  next_h = nn.CAddTable()({emb, next_h})
			 	else
			 	  next_h = nn.CAddTable()({x, next_h})
				end
		  end
			if dropout > 0 and L ~= n then
				next_h = nn.Dropout(dropout, nil, false):usePrealloc(nameL.."dropout",
																													{{opt.max_batch_l, input_size_L}})
																						 (next_h)
			end

			table.insert(outputs, next_c)
			table.insert(outputs, next_h)
		elseif opt.bow == 1 then
			table.insert(outputs, nn.CAddTable()({prev_c, prev_h}))
			table.insert(outputs, x)
		end
  end
  if model == 'dec' then
    local top_h = outputs[#outputs]
    local decoder_out
    local attn_output
    if opt.attn == 1 then
      local decoder_attn = make_decoder_attn(data, opt)
      decoder_attn.name = 'decoder_attn'
      if opt.guided_alignment == 1 then
        decoder_out, attn_output = decoder_attn({top_h, inputs[2]}):split(2)
      else
        decoder_out = decoder_attn({top_h, inputs[2]})
      end
    else
      decoder_out = nn.JoinTable(2)({top_h, inputs[2]})
      decoder_out = nn.Tanh()(nn.Linear(opt.rnn_size*2, opt.rnn_size, false)(decoder_out))
    end
    if dropout > 0 then
      decoder_out = nn.Dropout(dropout, nil, false):usePrealloc("dec_dropout",{RnnD})
                                                   (decoder_out)
    end
    table.insert(outputs, decoder_out)
    if opt.guided_alignment == 1 then
      table.insert(outputs, attn_output)
    end
  end
  return nn.gModule(inputs, outputs)
end

function make_decoder_attn(data, opt, simple)
  -- 2D tensor target_t (batch_l x rnn_size) and
  -- 3D tensor for context (batch_l x source_l x rnn_size)

  local inputs = {}
  table.insert(inputs, nn.Identity()())
  table.insert(inputs, nn.Identity()())
  local target_t = nn.Linear(opt.rnn_size, opt.rnn_size, false)(inputs[1])
  local context = inputs[2]
  simple = simple or 0
  -- get attention

  local attn = nn.MM():usePrealloc("dec_attn_mm1",
                                   {{opt.max_batch_l, opt.max_sent_l_src, opt.rnn_size},{opt.rnn_size, opt.rnn_size, 1}},
                                   {{opt.max_batch_l, opt.max_sent_l_src, 1}})
                      ({context, nn.Replicate(1,3)(target_t)}) -- batch_l x source_l x 1
  attn = nn.Sum(3)(attn)
  local softmax_attn = nn.SoftMax()
  softmax_attn.name = 'softmax_attn'
  attn = softmax_attn(attn)
  local attn_output
  if opt.guided_alignment == 1 then
    attn_output = attn
  end
  attn = nn.Replicate(1,2)(attn) -- batch_l x 1 x source_l

  -- apply attention to context
  local context_combined = nn.MM():usePrealloc("dec_attn_mm2",
                                               {{opt.max_batch_l, 1, opt.max_sent_l_src},{opt.max_batch_l, opt.max_sent_l_src, opt.rnn_size}},
                                               {{opt.max_batch_l, 1, opt.rnn_size}})
                                  ({attn, context}) -- batch_l x 1 x rnn_size
  context_combined = nn.Sum(2):usePrealloc("dec_attn_sum",
                                           {{opt.max_batch_l, 1, opt.rnn_size}},
                                           {{opt.max_batch_l, opt.rnn_size}})
                              (context_combined) -- batch_l x rnn_size
  local context_output
  if simple == 0 then
    context_combined = nn.JoinTable(2):usePrealloc("dec_attn_jointable",
                                                   {{opt.max_batch_l,opt.rnn_size},{opt.max_batch_l, opt.rnn_size}})
                                      ({context_combined, inputs[1]}) -- batch_l x rnn_size*2
    context_output = nn.Tanh():usePrealloc("dec_attn_tanh",{{opt.max_batch_l,opt.rnn_size}})
                              (nn.Linear(opt.rnn_size*2,opt.rnn_size,false):usePrealloc("dec_attn_linear",
                                                                                        {{opt.max_batch_l,2*opt.rnn_size}})
                                                                           (context_combined))
  else
    context_output = nn.CAddTable():usePrealloc("dec_attn_caddtable1",
                                                {{opt.max_batch_l, opt.rnn_size}, {opt.max_batch_l, opt.rnn_size}})
                                   ({context_combined,inputs[1]})
  end
  if opt.guided_alignment == 1 then
    return nn.gModule(inputs, {context_output, attn_output})
  else
    return nn.gModule(inputs, {context_output})
  end
end

function make_generator(data, opt)
  local model = nn.Sequential()
  model:add(nn.Linear(opt.rnn_size, data.target_size))
  model:add(nn.LogSoftMax())
  local w = torch.ones(data.target_size)
  w[1] = 0
  criterion = nn.ClassNLLCriterion(w)
  criterion.sizeAverage = false
  return model, criterion
end

-- cnn Unit
function make_cnn(input_size, kernel_width, num_kernels)
  local output
  local input = nn.Identity()()
  if opt.cudnn == 1 then
    local conv = cudnn.SpatialConvolution(1, num_kernels, input_size,
      kernel_width, 1, 1, 0)
    local conv_layer = conv(nn.View(1, -1, input_size):setNumInputDims(2)(input))
    output = nn.Sum(3)(nn.Max(3)(nn.Tanh()(conv_layer)))
  else
    local conv = nn.TemporalConvolution(input_size, num_kernels, kernel_width)
    local conv_layer = conv(input)
    output = nn.Max(2)(nn.Tanh()(conv_layer))
  end
  return nn.gModule({input}, {output})
end

function make_highway(input_size, num_layers, output_size, bias, f)
  -- size = dimensionality of inputs
  -- num_layers = number of hidden layers (default = 1)
  -- bias = bias for transform gate (default = -2)
  -- f = non-linearity (default = ReLU)

  local num_layers = num_layers or 1
  local input_size = input_size
  local output_size = output_size or input_size
  local bias = bias or -2
  local f = f or nn.ReLU()
  local start = nn.Identity()()
  local transform_gate, carry_gate, input, output
  for i = 1, num_layers do
    if i > 1 then
      input_size = output_size
    else
      input = start
    end
    output = f(nn.Linear(input_size, output_size)(input))
    transform_gate = nn.Sigmoid()(nn.AddConstant(bias, true)(
        nn.Linear(input_size, output_size)(input)))
    carry_gate = nn.AddConstant(1, true)(nn.MulConstant(-1)(transform_gate))
    local proj
    if input_size==output_size then
      proj = nn.Identity()
    else
      proj = nn.Linear(input_size, output_size, false)
    end
    input = nn.CAddTable()({
        nn.CMulTable()({transform_gate, output}),
        nn.CMulTable()({carry_gate, proj(input)})})
  end
  return nn.gModule({start},{input})
end

