# Third-Party Notices

The ProxyDiff NAS reproduction package includes or interoperates with the
following third-party material. Original copyright and license headers are
preserved in the source files.

## Bundled NB301 runtime

- NASLib and Samsung zero-cost NAS components: Apache License 2.0. The bundled
  license text is at `reproduction/nas/runtime/LICENSE-APACHE-2.0`.
- EPE-NAS measure (`predictors/pruners/measures/epe_nas.py`): MIT License,
  copyright Vasco Lopes. The full MIT notice is retained at the top of that
  file.
- Zen-NAS measure (`predictors/pruners/measures/zen.py`): copyright Alibaba
  Group Holding Limited; retained source notice and upstream Apache-2.0 terms
  apply.
- `utils/DownsampledImageNet.py`: copyright Xuanyi Dong; the original source
  notice is retained.

## External benchmark dependencies

- NAS-Bench-301 surrogate models are downloaded from the official
  `automl/nasbench301` release by the runtime and remain subject to their
  upstream terms.
- The bundled NB301 zero-cost benchmark payload and fixed architecture pool are
  research artifacts used for result reproduction. Their upstream terms and
  citations continue to apply.

The project license does not replace or restrict any third-party license.
