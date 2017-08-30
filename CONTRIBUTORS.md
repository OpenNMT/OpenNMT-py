PyOpenNMT is a community developed project and we love developer contributions. 

When sending a PR we will check the following: 

- Please ensure that there are no style issues. We run an automatic check that calls `flake8`. If that fails we cannot accept the PR.
- Please ensure that unittest is passed before sending PR. To run unittests, call `python -m unittest discover`.
- When modifying class constructor, please make the arguments as same naming style as its superclass in pytorch.
- There is also a basic model trained in using continuous integration. See `.travis.yml` for details.   
- If your change is based on a paper, please include a clear comment and reference in the code. 
- If your function takes/returns tensor arguments, please include assertions to document the sizes. See `GlobalAttention.py` for examples. 
