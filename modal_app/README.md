# Modal AI stages

All AI model download and inference lives here (SAM 3.1, ball, …).

**Do not** install model weights on the laptop or in `worker/`.

> Folder is named `modal_app/` (not `modal/`) so it does not shadow the
> Modal Python SDK package.

## Deploy

```powershell
$env:PYTHONUTF8='1'
modal deploy modal_app/app.py
# or: npm run modal:deploy
```

App: https://modal.com/apps/jamesyang663/main/deployed/volleyball-ai

Requires secret `huggingface` with `HF_TOKEN` and HF access to
[facebook/sam3](https://huggingface.co/facebook/sam3) / sam3.1.

## Local worker

```powershell
# .env
USE_MOCK_TRACKS=0
SAM3_PROMPT=person

.\.venv\Scripts\pip install modal
npm run worker
```

Functions:
- `track_players` — SAM 3.1 text-prompted video tracking (`person` by default)
- `track_ball` — motion-blob ball tracker (placeholder until a dedicated ball model)
