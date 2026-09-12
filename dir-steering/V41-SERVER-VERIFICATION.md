# ABL-052 HTTP server verification

Date: 2026-09-12. Platform: Apple M1 Ultra, 128 GiB, macOS Metal,
SSD streaming. This adds server startup/session wiring to the reviewed ABL-049
CLI implementation and the existing W1 intervention APIs; projection math is
unchanged.

## Model-free checks

- `make ds4-server ds4_test ds4_server_cpu.o`: passed. The CPU object compiles,
  but V4.1 steering correctly requires Metal; no CPU inference was run.
- `make test-v41-server test-v41-cli`: passed. Coverage includes explicit finite
  strength, both sites, missing/invalid files, legacy defaults and explicit
  scales, rejected disk caches and unsupported modes, model-digest mismatch,
  failed hashing and configuration cleanup. The session factory configures
  every tested slot exactly once.
- `./ds4_test --server`: existing parser, rendering and cache regressions passed.
- Fresh read-only code review: `ALL GOOD`.
- Full `make test`: the early gates passed, but the session-snapshot test stopped
  at the normal instance lock held by the coordinated benchmark. This was not
  a full gate pass, and the lock was not bypassed. A final retry after the HTTP
  processes exited again passed the early gates, then stopped because the
  default `ds4flash.gguf` is absent. No substitute model was used for those
  V4-specific tests.

## Live HTTP smoke

The separate research task explicitly released the shared memory window before
these serial runs. The normal instance lock remained enabled. The model was
reused in place and no second model process ran concurrently.

Deployment SHA256:
`1ce6a8f8806205c13330d7ca287bd198331dc5ca35ccc5d8a9a92a188a6f6f42`.
Direction: `dealignai-writer-proxy.ds41dir`, writer layers 10–36,
SHA256 `451ca636f821c9416e40439c9eaa1face7161d3e6a26599cf23b468de592e4b7`.
The direction comes from the packaged W3 native-weight-difference proxy, not
the synthetic e0 fixtures used in ABL-049.

Reproduce from the runtime checkout after coordinating model memory:

```sh
python3 tests/test_v41_server_smoke.py \
  ../DeepSeek-V4.1-Flash-Q2.gguf \
  ../DeepSeek-V4.1-Flash-Abliteration-Adapter/dealignai-writer-proxy.ds41dir
```

The script uses three fresh server processes: stock, alpha 0 and alpha 1,
512-token context, greedy generation, reasoning disabled and a 16-token limit.
Each server answers an ordinary non-streaming arithmetic request, repeats that
prompt as SSE streaming, then receives a different arithmetic prompt. It checks
valid model discovery, completion/finish markers, identical streaming text and
exact stock/zero output parity. Each steered startup verifies the opened model
SHA256 once before creating its session.

All nine standard HTTP requests and two additional conversation requests passed. Each response was `42`; normal and streaming
outputs matched, and all zero-strength outputs were identical to stock.
Both steered logs record one successful full-model verification before listening.
The additional dealignai conversation reused 19 live KV tokens and returned `42`
on both turns. The standard repeated completed prompt caused invalidation,
which is distinct from this verified continuation cache hit.

| Arm | Three response texts | Seconds including startup and requests |
|---|---|---:|
| stock | 42 / 42 / 42 | 14.64 |
| zero | 42 / 42 / 42 | 225.86 |
| steered | 42 / 42 / 42 | 238.45 |

Local raw response bodies and runtime logs are retained under
`work/v41-server-smoke/`; `results.json` records the per-arm results.


## Limits

This is integration evidence, not refusal-rate or capability qualification.
HTTP live testing uses one session slot and the dealignai writer proxy.
Both sites and multiple slot configuration are covered by focused fixtures;
ABL-049 separately exercised both sites on the real model. Concurrent HTTP load,
non-SSD native GPU batching, CUDA and distributed execution were not qualified.
Disk KV persistence is explicitly rejected because its identity omits steering;
live KV reuse is supported with immutable startup settings.
