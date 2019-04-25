local function get_ngrams(s, n, count)
   local ngrams = {}
   count = count or 0
   for i = 1, #s do
      for j = i, math.min(i+n-1, #s) do
    local ngram = table.concat(s, ' ', i, j)
    local l = j-i+1 -- keep track of ngram length
    if count == 0 then
       table.insert(ngrams, ngram)
    else
       if ngrams[ngram] == nil then
          ngrams[ngram] = {1, l}
       else
          ngrams[ngram][1] = ngrams[ngram][1] + 1
       end
    end
      end
   end
   return ngrams
end

local function get_ngram_prec(cand, ref, n)
   -- n = number of ngrams to consider
   local results = {}
   for i = 1, n do
      results[i] = {0, 0} -- total, correct
   end
   local cand_ngrams = get_ngrams(cand, n, 1)
   local ref_ngrams = get_ngrams(ref, n, 1)
   for ngram, d in pairs(cand_ngrams) do
      local count = d[1]
      local l = d[2]
      results[l][1] = results[l][1] + count
      local actual
      if ref_ngrams[ngram] == nil then
    actual = 0
      else
    actual = ref_ngrams[ngram][1]
      end
      results[l][2] = results[l][2] + math.min(actual, count)
   end
   return results
end

--function get_bleu(cand, ref, n)
function get_bleu(cand, ref, n, score, attn)
   n = n or 4
   local m = 1
   if type(cand) ~= 'table' then
      cand = cand:totable()
   end
   if type(ref) ~= 'table' then
      ref = ref:totable()
   end
   local r = get_ngram_prec(cand, ref, n)
--   print(r)
   local bp = math.exp(1-math.max(1, #ref/#cand))
   local correct = 0
   local total = 0
   local bleu = 1
   for i = 1, n do
      if r[i][1] > 0 then
	 if r[i][2] == 0 then
	    m = m*0.5
	    r[i][2] = m
	 end
	 local prec = r[i][2]/r[i][1]
	 bleu = bleu * prec
      end      	 --      correct = correct + r[i][2]
--      total = total + r[i][1]
   end
   bleu = bleu^(1/n)
   return bleu*bp
end
