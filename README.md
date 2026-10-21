# industrial-ai-platform

Industrial AI platform backend, built with FastAPI and managed by [uv](https://docs.astral.sh/uv/).

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/)

From-scratch environment setup (Windows: uv, git, GitHub connectivity) is documented in
**[docs/SETUP.md](docs/SETUP.md)**.

The frozen HTTP contract for the extraction API lives in **[docs/api-spec.md](docs/api-spec.md)**.

## Setup

```powershell
uv sync
```

`uv sync` creates `.venv` and installs both runtime and dev dependencies.

## Run

```powershell
uv run uvicorn app.main:app --reload
```

Then open:

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health check: <http://127.0.0.1:8000/health>

## Verify

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected output:

```json
{"status":"ok"}
```

Extraction endpoint:

```powershell
curl.exe -s -X POST http://127.0.0.1:8000/api/v1/extract `
  -H "Content-Type: application/json" `
  --data-binary "@body.json"     # {"text":"合同编号：HT-2026-001，甲方：张三","fields":["合同编号","甲方"]}
```

Expected output:

```json
{"request_id":"…","data":{"合同编号":"HT-2026-001","甲方":"张三"},"confidence":1.0,"elapsed_ms":101}
```

> Windows curl notes: use `curl.exe` (not `curl`), and pass JSON via `--data-binary "@file"` —
> PowerShell mangles inline JSON quotes. See `docs/SETUP.md` for details.

> On Windows PowerShell, `curl` is an alias for `Invoke-WebRequest`. Always use `curl.exe` to get real curl behaviour.

## Development

```powershell
uv run ruff check .        # lint
uv run ruff format .       # format
uv run pytest --cov=app    # tests + coverage
uv run pre-commit install  # enable git hooks (run once after git init)
```

## Contract stability

`POST /api/v1/extract` is **v1-frozen**. The rule-based extractor in
`app/services/extractor.py` is a placeholder: W2 swaps it for an LLM implementation behind the
same `async extract(text, fields)` signature, and `tests/test_api.py` must stay green **unchanged**.
That test file is the executable form of `docs/api-spec.md`.

## License

MIT
