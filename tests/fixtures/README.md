# Scientific fixture provenance

`v1_scientific_fixture.csv` contains five complete eight-mode states copied without
recalculation from the academic V1 `data/dataset_real.csv`. It is deliberately small,
reviewable, and suitable for deterministic tests; it is not a replacement training or
validation dataset.

- Academic repository commit: `233e28634a70e1e0898ced684772f2b0e57502cb`
- Parent dataset SHA-256: `0040d1df643af60b332464407384a259b0231af277d17a1e09efb91f87fc4c85`
- Selected state triples `(elevation_deg, abs(v_rel_kmps), n_nodes)`:
  `(5, 0, 10000)`, `(14, 5.5, 50000)`, `(27, 2.5, 10000)`,
  `(45, 4, 50000)`, `(85, 0.5, 100000)`.

The product repository now also contains the separately versioned, LF-canonical real pooled
dataset used by the production k-NN adapter:

- path: `packages/meseleo-core/src/meseleo_core/data/v1_pooled_dataset.csv`;
- version: `v1-pooled-real-1`;
- SHA-256: `bbd0c4518c74e33ee9538be1483adc35f1f56b3a25b3b7ac4c74765777c51053`;
- shape: 1,505 complete states and 12,040 data rows.

That packaged dataset does not change this file's purpose: `v1_scientific_fixture.csv` remains the
small deterministic regression/demo fixture and must never be presented as the production dataset.
Production Chroma stores and downloaded models remain excluded from the repository and review
archives.
