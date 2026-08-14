# ProxyDiff

ProxyDiff is a proxy-guided differentiable scoring method for structural
component selection.

## Reproduction Code

The current public reproduction package contains the NB301 NAS pipeline and
clean analysis code:

```text
reproduction/nas/
reproduction/nas_reported/
```

See `reproduction/nas/README.md` for the end-to-end score, cache,
refinement, and evaluation commands. See `reproduction/nas_reported/README.md`
for component, geometry, calibration, and subset analyses.

Check the public reproduction package layout:

```bash
python reproduction/check_package.py
```

ProxyDiff is released under the Apache License 2.0. Bundled third-party
components and research assets are documented in `THIRD_PARTY_NOTICES.md`.
