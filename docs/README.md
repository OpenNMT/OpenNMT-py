[MkDocs](http://www.mkdocs.org/) is used to generate the documentation at http://opennmt.net/OpenNMT/.

If you want to visualize and deploy the documentation, continue reading the next sections.

## Installation

```bash
pip install mkdocs mkdocs-material python-markdown-math
```

## Workflow

1. Edit the Markdown documentation in `docs/`
2. Visualize the documentation locally with `mkdocs serve`
3. Commit your documentation changes
4. Generate and deploy the static website on the `gh-pages` branch with `mkdocs gh-deploy` (if you are testing on a fork, don't forget to configure the remote with the `-r` option)

## Tips

### Adding pages

Update the main configuration file `mkdocs.yml`.

### Generating options listing

```bash
./docs/options/generate.sh
```
