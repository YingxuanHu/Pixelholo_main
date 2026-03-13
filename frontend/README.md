# PixelHolo Voice Studio (Frontend)

This frontend is the control panel for the local StyleTTS2 workflow.
It talks to the FastAPI backend in `voice_cloning`.

## Run

1) Start the backend:

```bash
cd voice_cloning
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

2) Install frontend deps + start:

```bash
cd frontend
npm install
npm run dev
```

Open: `http://127.0.0.1:5173`

## Backend Endpoints Used

- `POST /upload` (multipart)
- `POST /preprocess` (stream logs)
- `POST /train` (stream logs)
- `POST /stream` (NDJSON streaming audio chunks)
- `GET /profiles` (list existing profiles)

The UI plays audio as soon as the first chunk arrives.
