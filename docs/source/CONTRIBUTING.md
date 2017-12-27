# Contributors

OpenNMT-py is a community developed project and we love developer contributions.

Before sending a PR, please do this checklist first:

- Please run `tools/pull_request_chk.sh` and fix any errors. When adding new functionality, also add tests to this script. Included checks:
    1. flake8 check for coding style;
    2. unittest;
    3. continuous integration tests listed in `.travis.yml`.
- When adding/modifying class constructor, please make the arguments as same naming style as its superclass in pytorch.
- If your change is based on a paper, please include a clear comment and reference in the code. 
- If your function takes/returns tensor arguments, please include assertions to document the sizes. See `GlobalAttention.py` for examples. 
