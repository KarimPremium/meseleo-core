# meseleo-core

Framework-independent decision core for reliability-constrained link adaptation on
direct-to-satellite IoT, LEO and non-terrestrial 6G links.

The package selects a radio mode (LoRa SF7 to SF12, LR-FHSS DR8 and DR9) for a given
link geometry and operational requirement, and returns an auditable trace of why that
mode was selected. It contains no web framework, no database and no modem executor.

## Design rule

Prediction and decision are separate stages, and they are separated on purpose.

A learned component estimates link performance. A deterministic operator, not the
learned component, decides whether a transmission is admissible under the declared
constraints. The failure mode this guards against was measured on the research work
this package derives from: the generative model predicted well and decided badly,
independently of model size. Generation predicts, an exact operator decides.

Consequences visible in the API:

- Constraint evidence is tri-state. A constraint is satisfied, violated, or lacking
  the evidence needed to conclude. Missing evidence is never silently treated as
  satisfaction.
- Confidence is reported as unavailable rather than invented when the versioned
  predictor provides no calibrated interval. Out-of-distribution status is not a
  substitute for a confidence interval.
- Transmission eligibility is separate from analysis. A non-visible geometry can be
  analysed, but it cannot produce an actionable command.
- The exact engine retains predictor provenance per radio mode and never calls an
  explanation or generative provider.

## Scope

Included in this repository:

| Module | Role |
| --- | --- |
| `models` | Immutable Pydantic contracts for scenarios, requests and results |
| `geometry` | Fixed-TLE SGP4 pass sampling: elevation, azimuth, range, radial speed, Doppler |
| `prediction` | Pooled-label k-NN adapter, optional lazy RAG adapter, predictor registry |
| `policy` | Compilation of operational requests into hard and soft constraints |
| `decision` | Deterministic reliability-constrained mode selection and audit trace |
| `pipeline` | Composition of the stages above |
| `commands` | Validated AT command string generation |
| `explanations` | Template-based post-decision explanation, outside the decision path |
| `research` | Predictor comparison harness |
| `datasets` | Dataset locations, provenance constants and integrity digests |
| `interfaces`, `configuration`, `exceptions`, `integration_errors` | Typed protocols, configuration and error taxonomy |

Not included, and not planned for this repository: the hosted Mission Studio
application, the transport API, tenancy, licensing, metering, billing adapters,
database migrations and production infrastructure. Those belong to the hosted
service.

## Data

The pooled evaluation dataset `v1-pooled-real-1` (1,505 states, 12,040 rows, eight
radio modes) and its held-out benchmark set are not distributed here. They are
available under a separate commercial licence. `datasets.py` keeps their version
string, row counts and SHA-256 digests so that a licensed copy can be verified
byte-for-byte against the provenance this package expects.

Without a licensed dataset in place, `V1Dataset` raises `DatasetNotFoundError` rather
than falling back to anything. That is deliberate: silently substituting a smaller
dataset would produce numbers that look like results and are not.

What ships in this repository is the reduced five-state fixture used by the test
suite, at `tests/fixtures/v1_scientific_fixture.csv`. It exercises every stage of the
pipeline end to end and is sufficient to run the full suite. It is a test fixture, not
evaluation data, and figures obtained from it must never be reported as evaluation
results.

## Install

Python 3.12 is required.

```bash
pip install -e ".[dev]"
```

The optional RAG adapter pulls Chroma and Ollama dependencies and is not installed by
default. Its absence does not disable k-NN prediction.

```bash
pip install -e ".[dev,rag]"
```

## Verify

```bash
pytest
ruff check .
mypy src
```

## Units

PER is a ratio in `[0, 1]`. Goodput is bit/s. Packet-window capacity is packets per
window. Expected delivery latency and airtime are ms. Energy per attempt is mJ.
Doppler is Hz. Radial speed is km/s.

## Limits

- The pooled dataset is scientific evidence for the conditions it covers, not proof
  of performance for an arbitrary satellite, terminal, geography, channel plan or
  payload.
- Pass acquisition, loss of signal and culmination are bounded by the requested
  sampling interval.
- SNR and RSSI inputs are not treated as k-NN features when the versioned predictor
  does not use them.
- A generated AT command is a validated string. No physical modem is contacted.

## Origin

This package derives from research on AI-driven modulation selection for LEO
direct-to-satellite IoT carried out at ESIEE Paris, described in a paper submitted to
an IEEE conference. The research code, dataset and evaluation harness of that work are
published separately under the MIT licence:

https://github.com/houcinemessad/Modulation_AIAgent_for_LEO_-LRFHSS-LoRa-

See `NOTICE.md` for attribution details.

## Licence

GNU Affero General Public License v3.0 or later. See `LICENSE`.

The AGPL applies to network use: an operator that exposes a modified version of this
core as a network service is required to offer the corresponding source of that
modified version to its users.

A separate commercial licence, which removes that obligation and covers the pooled
evaluation dataset, is available for embedding this core in a closed product or
service. Contact: houcinemessad@gmail.com

## Contributing

Contributions require a signed Contributor Licence Agreement. See `CONTRIBUTING.md`
and `CLA.md`.
