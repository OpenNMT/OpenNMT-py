local beam = require 's2sa.beam'

function main()
  beam.init(arg)
  local opt = beam.getOptions()
	local typeOfInput = opt.input_type
  if opt.interactive_mode == 0 then
		assert(path.exists(opt.src_file), 'src_file does not exist')
		local sent_id = 0
		file_size = 0
		for _ in io.lines(opt.src_file) do
			file_size = file_size + 1
		end
		-- produce anonymized version of AMR file and the corresponding alignments in two separate files
		if typeOfInput ~= 'anonymized' and typeOfInput ~= 'textAnonymized' then
			anonymizeFile(typeOfInput, opt.src_file)
		end
		local file = io.open(opt.src_file .. '.anonymized', "r")
		local out_file = io.open(opt.output_file .. '.pred.anonymized','w')
		for line in file:lines() do
			sent_id = sent_id + 1
			xlua.progress(sent_id, file_size)
			if typeOfInput == 'text' or typeOfInput == 'textAnonymized' then
				out_file:write(parse('textAnonymized', line, opt.verbose) .. '\n')
			else
				out_file:write(generate('anonymized', line, opt.verbose) .. '\n')
			end
		end
		out_file:close()
		file:close()
		-- deAnonymize predictions using alignments
		if typeOfInput ~= 'anonymized' and typeOfInput ~= 'textAnonymized' then
			deAnonymizeFile(typeOfInput, opt.src_file)
		end
		if typeOfInput == 'text' then
			killNerServer()
		end
	else
		if typeOfInput == 'text' or typeOfInput == 'textAnonymized' then
			print('Input text [Type q to exit]:')
		else
			print('Input AMR in ' .. typeOfInput .. ' format [Type q to exit]:')
		end
		while true do
			local input = io.read()
			if input == 'q' then
				if typeOfInput == 'text' then
					killNerServer()
				end
				break
			end
			if typeOfInput == 'text' or typeOfInput == 'textAnonymized' then
				print(parse(typeOfInput, input, opt.verbose))
			else
				print(generate(typeOfInput, input, opt.verbose))
			end
		end
	end
end

function generate(typeOfInput, input, verbose)
	if typeOfInput == 'anonymized' then
			local pred, pred_score, attn, pred_out = predSingleSentence(input)
			return pred_out
	else
			-- clean
			input = clean(input)
			-- anonymize
      local result, anonymizedInput, alignments = anonymizeAmr(typeOfInput, input, verbose)
      if not result then
      	return anonymizedInput -- error message
      else
      	-- predict
				local pred, pred_score, attn, pred_out = predSingleSentence(anonymizedInput)
				pred_out = stringx.replace(pred_out, '\"', '\\"')
				if verbose > 0 then
					print('predicted (anonymized): ' .. pred_out)
				end
				-- deAnonymize
		    return deAnonymize(pred_out, alignments, true)
			end
	end
end

function parse(typeOfInput, input, verbose)
	if typeOfInput == 'textAnonymized' then
			local pred, pred_score, attn, pred_out = predSingleSentence(input)
			return pred_out
	else
			-- clean
			input = clean(input)
			-- anonymize
      local anonymizedInput, alignments = anonymizeText(input, verbose)
			-- predict
			local pred, pred_score, attn, pred_out = predSingleSentence(anonymizedInput)
			pred_out = stringx.replace(pred_out, '\"', '\\"')
			if verbose > 0 then
				print('predicted (anonymized): ' .. pred_out)
			end
			-- deAnonymize
			output, _ = unpack(stringx.split(deAnonymize(pred_out, alignments, false), '#'))
			return output
	end
end

function clean(input)
	local flatInput = stringx.replace(input, '\n', ' ')
	flatInput = stringx.replace(flatInput, '\"', '\\"')
	return flatInput
end

function anonymizeFile(typeOfInput, path)
	local action
	if typeOfInput == 'stripped' then
		action = 'anonymizeAmrStripped'
	elseif typeOfInput == 'full' then
		action = 'anonymizeAmrFull'
	else 
		action = 'anonymizeText'
	end
	local f = io.popen('./anonDeAnon_java.sh ' .. action .. ' true \"' .. path .. '\"', w)
	f:close()
end

function anonymizeAmr(typeOfInput, input, verbose)
	-- anonymize and grab alignments
	local action
	if typeOfInput == 'stripped' then
		action = 'anonymizeAmrStripped'
	else
		action = 'anonymizeAmrFull'
	end
	local f = io.popen('./anonDeAnon_java.sh ' .. action .. ' false \"' .. input .. '\"', rw)
	local anonymizedInput, alignments = unpack(stringx.split(f:read('*all'), '#'))
	alignments = stringx.replace(alignments, '\n', '')
	if verbose > 0 then
		print('anonymized: ' .. anonymizedInput)
		print('alignments: ' .. alignments)
	end
	f:close()
	if anonymizedInput == 'FAILED_TO_PARSE' then
		if alignments == 'Failed to parse.' then
			return false, 'Failed to parse.', ""
		else
			return false, 'Failed to parse. ' .. alignments, ""
		end
	else
		return true, anonymizedInput, alignments
	end
end

function deAnonymizeFile(typeOfInput, path)
	local action
	if typeOfInput == 'text' then
		action = 'deAnonymizeAmr'
	else
		action = 'deAnonymizeText'
	end
	local f = io.popen('./anonDeAnon_java.sh ' .. action .. ' true \"' .. path .. '\"', w)
	f:close()
end

function deAnonymize(pred_out, alignments, isText)
	local f
	if alignments == '\n' then
		if isText then
			f = io.popen('./anonDeAnon_java.sh deAnonymizeText false \"' .. pred_out .. '\"', rw)
		else
			f = io.popen('./anonDeAnon_java.sh deAnonymizeAmr false \"' .. pred_out .. '\"', rw)
		end
	else
		if isText then
			f = io.popen('./anonDeAnon_java.sh deAnonymizeText false \"' .. pred_out .. '#' .. alignments .. '\"', rw)
		else
			f = io.popen('./anonDeAnon_java.sh deAnonymizeAmr false \"' .. pred_out .. '#' .. alignments .. '\"', rw)
		end
	end
	local deAnonymized = f:read('*all')
	deAnonymized = stringx.replace(deAnonymized, '\n', '')
	f:close()
	return deAnonymized
end

function deAnonymizeAmr(pred_out, alignments)

end

function anonymizeText(input, verbose)
	-- anonymize and grab alignments
	local f = io.popen('./anonDeAnon_java.sh anonymizeText false \"' .. input .. '\"', rw)
	local anonymizedInput, alignments = unpack(stringx.split(f:read('*all'), '#'))
	alignments = stringx.replace(alignments, '\n', '')
	if verbose > 0 then
		print('anonymized: ' .. anonymizedInput)
		print('alignments: ' .. alignments)
	end
	f:close()
	return anonymizedInput, alignments
end

function killNerServer()
	local f = io.popen('./anonDeAnon_java.sh anonymizeText false \"terminate_server\"', w)
	f:close()
end

main()
