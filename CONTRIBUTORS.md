PyOpenNMT is a community developed project and we love developer contributions. 

When sending a PR we will check the following: 

- Please unsure that there are no style issues. We run an automatic check that calls `flake8`. If that fails we cannot accept the PR.
- If your change is based on a paper, please include a clear comment and reference in the code. 
- If your function takes/returns tensor arguments, please include assertions to document the sizes. See `GlobalAttention.py` for examples. 
