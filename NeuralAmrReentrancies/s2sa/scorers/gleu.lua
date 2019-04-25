-- implementation of GLEU as defined in https://arxiv.org/abs/1609.08144
local function get_ngrams(s, maxn)
  local ngrams = {}
  local size = 0
  for n = 1, maxn do
    for i = 1, #s do
      for j = i, math.min(i+n-1, #s) do
        local ngram = table.concat(s, ' ', i, j)
        if not ngrams[ngram] then
          ngrams[ngram]=1
          size=size+1
        end
      end
    end
  end
  return size,ngrams
end

function get_gleu(cand, ref, n)
  n = n or 4
  local ncand, ngrams_cand = get_ngrams(cand, n)
  local nref, ngrams_ref = get_ngrams(ref, n)

  local count_match = 0
  for v,_ in pairs(ngrams_cand) do
    if ngrams_ref[v] then
      count_match = count_match + 1
    end
  end
  return math.min(count_match/nref, count_match/ncand)
end
