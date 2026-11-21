# Conformal Quantile Calibration

SuperLab V27 FAIR4RS Release V3 · Stacked Cheese Intelligence LLC.

Nineteen quantiles, PAVA projection, chronological conformal evaluation and sequential e-processes. The 184 to 0 crossings figure is from the separate 36,040-row historical reference.

[Workbench](https://stackedcheese.com/research) · [Umbrella repository](https://github.com/stackedcheese/superlab-v27-open-science-benchmarks) · [Author ORCID](https://orcid.org/0009-0007-7479-7082)

## Reproduction scope

The script generates the chronological train, calibration and distribution-shift fixtures. Use Python 3.11 with the pinned CPU dependencies. The historical crossing numbers are reference evidence in the result and require their original source to be reproduced.

Run in the applicable pinned environment using a fresh output filename.

```sh
python reproduce.py --output replay_result.json
```

## Result and review commitments

- result.json SHA-256 `50126a7ae7410075cef5c6932f76af37feb7b5ee59ee199ed19a2d2b2a024ba2`
- Claude Ring 0 PASS, aggregate-capsule review SHA-256 `be071e8ae668eb9ac36921839e1d12a2347e4e04165b123ceb319d177833d36b`
- Original public source [track2_calibration at 7efceed48955](https://github.com/stackedcheese/superlab-v27-open-science-benchmarks/tree/7efceed48955be0cfc50ea3751c33ebd2417d9ff/track2_calibration)

The review applies to the aggregate capsule. Use the result file and reproduction scope to interpret the benchmark. The MIT license covers the included software. Original datasets and source assets retain their own terms.

## Citation

Levord Rose Jr. (2026). SuperLab V27 Open-Science Benchmark Suite, Release V3. Stacked Cheese Intelligence LLC. https://github.com/stackedcheese/superlab-v27-open-science-benchmarks/releases/tag/v3.0.0. ORCID https://orcid.org/0009-0007-7479-7082.

Machine-readable citation is in [CITATION.cff](CITATION.cff).
