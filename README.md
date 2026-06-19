# ProxyDiff

ProxyDiff is a proxy-guided differentiable scoring method for structural
component selection.

## Reproduction Code

The current public reproduction package contains the NB301 NAS pipeline and the
reported-result analysis scripts:

```text
reproduction/nas/
reproduction/analysis/
```

See `reproduction/nas/README.md` for the end-to-end score, cache, refinement,
and evaluation commands.  See `reproduction/analysis/README.md` for the figure
and reported-result checking scripts.

Check the public reproduction package layout:

```bash
python reproduction/check_package.py
```
