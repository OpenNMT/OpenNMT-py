require('json')
require('socket')

local beam = require 's2sa.beam'
local debugger = require('fb.debugger')

function main()
	beam.init(arg)
	local opt = beam.getOptions()

	server = assert(socket.bind("*", opt.port))
	server_state = "running"
	ip, port = server:getsockname()

	while server_state == "running" do
		print("wait for connection on port: " .. opt.port)
		local client = server:accept()
		local input, err = client:receive()
		local jsonInput = json.decode(input)
		local flatInput = stringx.replace(jsonInput['input'], '\n', ' ')
		flatInput = stringx.replace(flatInput, '\"', '\\"')
		if (opt.verbose > 0) then
			print('user: ' .. jsonInput['typeOfAmr'] .. '=' .. flatInput)
		end
		local action = jsonInput['typeOfAmr']
		if action == 'full' then
			action = 'anonymizeAmrFull'
		elseif action == 'stripped' then
			action = 'anonymizeAmrStripped'
		else
			action = 'anonymizeText'
		end
		-- anonymize and grab alignments
		local f = io.popen('./anonDeAnon_java.sh ' .. action .. ' false  \"' .. flatInput .. '\"', rw)
		local anonymizedInput, alignments, graph
		if action == 'anonymizeText' then
			anonymizedInput, alignments = unpack(stringx.split(f:read('*all'), '#'))
		else
			anonymizedInput, alignments, graph = unpack(stringx.split(f:read('*all'), '#'))
		end
		alignments = stringx.replace(alignments, '\n', '')
		if (opt.verbose > 0) then
			print('anonymized: ' .. anonymizedInput)
			print('alignments: ' .. alignments)
			if (graph) then
				print('graph: ' .. graph)
			end
		end
		f:close()
		if anonymizedInput == 'FAILED_TO_PARSE' then
			if alignments == 'Failed to parse.' then
				client:send('Failed to parse.\n')
			else
				client:send('Failed to parse. ' .. alignments .. '\n')
			end
		else
			local pred, pred_score, attn, pred_out = beam.predSingleSentence(anonymizedInput)
			pred_out = stringx.replace(pred_out, '\"', '\\"')
			if (opt.verbose > 0) then
				print('predicted: ' .. pred_out)
			end
			-- deAnonymize (always do the opposite action of what was given as input
			if action == 'anonymizeText' then
				action = 'deAnonymizeAmr'
			else
				action = 'deAnonymizeText'
			end
			if alignments == '\n' then
				f = io.popen('./anonDeAnon_java.sh ' .. action .. ' false \"' .. pred_out .. '\"', rw)
			else
				f = io.popen('./anonDeAnon_java.sh ' .. action .. ' false \"' .. pred_out .. '#' .. alignments .. '\"', rw)
			end
			local deAnonymized
			if action == 'deAnonymizeAmr' then
				deAnonymized, graph = unpack(stringx.split(f:read('*all'), '#'))
			else
				deAnonymized = f:read('*all')
			end
			deAnonymized = stringx.replace(deAnonymized, '\n', '')
			deAnonymized = stringx.replace(deAnonymized, '\"', '\\\"')
			f:close()
			if opt.verbose > 0 then
				print('deAnonymized: ' .. deAnonymized)
				print('graph: ' .. graph)
			end
			client:send('{\"predText\":\"' .. deAnonymized .. '\",' .. graph .. '}\n')
		end
		client:close()
	end
end

main()
