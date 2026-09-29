# cuAttest documentation

**[How it works — illustrated](index.html)** is the best starting point: the
two-process split, the hash tree and the statement graph, drawn. Open it in a
browser (`open docs/index.html`); GitHub shows HTML as source, so it will not
render inline here.

Then:

- [Quickstart](quickstart.md) — install, build the kernel, run it
- [Multi-GPU](multi-gpu.md) — distributed models, device routing, and verification
- [How it works](how-it-works.md) — the same mechanism, as text
- [Expected CID](expected-cid.md) — compare submitted VRAM spans with a checkpoint
- [The kernel](kernel.md) — a breakdown of the CUDA module itself
- [API](api.md) — HTTP and Python reference
- [Trust model](trust-model.md) — what a signature here does and does not prove
- [Operations](operations.md) — deployment, containers, troubleshooting
- [Performance](performance.md) — GPT-2 baselines and a near-capacity 397B-model benchmark
- [Sanitizers](sanitizers.md) — CPU/CUDA audit results, fixes, limitations, and repeatable checks
- [Sanitizer report — 2026-09-08](testing/sanitizers-2026-09-08.md) — full audit matrix, diagnostics, and coverage limits

## Publishing

`.github/workflows/pages.yml` deploys this directory to GitHub Pages on every
push that touches it. Enable it once, in **Settings → Pages → Source →
"GitHub Actions"**; until then the deploy step fails with a 404.

The site root is `index.html`. The markdown files are written to be read on
GitHub, where they render; Pages serves them as plain files.
