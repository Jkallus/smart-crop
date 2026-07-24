# smart-crop

Automates producing per-aspect-ratio crops of landscape photos, replacing a manual Lightroom
workflow. A locally-hosted vision LLM looks at each full-resolution export, decides which of five
target aspect ratios are worth producing, and picks a crop for each; deterministic Python does the
actual pixel math and file output.

See `agent_spec.md` for the full design (crop-box math, tool schema, prompt guidance, known
gotchas) and `plan.md` for the original problem statement.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a locally-hosted OpenAI-compatible vision model
server (developed against oMLX serving Qwen3 and Gemma on Apple Silicon).

```bash
uv sync
export OMLX_API_KEY=<your key>
# export OMLX_BASE_URL=http://127.0.0.1:8000/v1  # only if not the default
```

## Target ratios

| Name | Ratio | Use case |
|---|---|---|
| `tv` | 16:9 | TVs, monitors |
| `macbook` | 16:10 | MacBook Pro screen |
| `ultrawide` | 5120:2160 | Ultrawide monitor |
| `ipad` | 4:3 | iPad screen |
| `iphone` | 9:19.5 | iPhone portrait |

## Usage

### Single image or folder, one model

```bash
uv run python -m smart_crop.agent <image_or_folder> <output_dir> --backend qwen
```

Prints each target's decision and writes crops into `<output_dir>/<target>/`.

### Compare multiple models over a batch, with review flags

```bash
uv run python -m smart_crop.compare <image_or_folder> <output_dir> <log.jsonl> --backends qwen gemma
```

Dispatches all (image, backend) jobs concurrently, writes one crop per backend into
`<output_dir>/<backend>/<target>/`, and logs one JSON line per decision to `<log.jsonl>` --
including cheap geometry-only "worth a look" flags (`flags.py`) and cross-backend disagreement
flags, so a batch of hundreds of images doesn't require eyeballing every crop.

Add a backend by name in `backends.py` (needs a `base_url`/`api_key` env var pair and a model
name); both currently point at the same oMLX server.

### Manual/hand-written decisions (no model)

```bash
uv run python -m smart_crop.batch <image_dir> <decisions.json> <output_dir>
```

Useful for testing the crop math in isolation. See `smart_crop/batch.py` docstring for the JSON
format.

## Tests

```bash
uv run pytest
```
