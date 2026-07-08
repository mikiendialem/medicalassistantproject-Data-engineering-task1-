# Symptom Chart — Triage Assistant (Prototype)

A classical ML symptom checker: pick symptoms, get ranked *possible* conditions
with confidence scores. **This is an educational prototype, not a diagnostic
tool.** See disclaimers below and keep them in the app if you deploy it further.

## What's in here

```
medical-agent/
├── data/
│   ├── training_data.csv     # 132 symptoms x 41 diseases (public demo dataset)
│   └── test_data.csv         # held-out evaluation set
├── model/
│   ├── train.py              # trains the Random Forest classifier
│   ├── disease_classifier.joblib   # trained model
│   ├── symptom_list.json     # the 132 symptom keys the model expects
│   └── disease_list.json     # the 41 disease labels
├── backend/
│   ├── main.py                # FastAPI app: /symptoms, /predict, /health
│   └── requirements.txt
└── frontend/
    └── index.html             # standalone UI, calls the backend API
```

## Run it locally

### Quick start

1. Open a terminal in the project root.
2. Install the backend dependencies:

```bash
cd backend
pip install -r requirements.txt
```

3. If you need the optional GitHub/Infermedica proxy, copy `.env.example` to
  `.env` and fill in `INFERMEDICA_APP_ID` and `INFERMEDICA_APP_KEY`.

4. Start the backend server:

```bash
python -m uvicorn main:app --reload --port 8420
```

5. Open `http://localhost:8420/` in your browser.
6. Type symptoms, choose the matching chips, and click the check button.

If you are on Windows, you can run the helper script from the project root
instead:

```bat
scripts\start_app.bat
```

```bash
cd backend
pip install -r requirements.txt
python -m uvicorn main:app --reload --port 8420
```

Then open `http://localhost:8420/` in a browser. The FastAPI app serves the
frontend and API from the same origin, so you do not need a second static file
server anymore.

If you want to point the frontend at a different backend later, edit
`frontend/config.json` and change `apiUrl`.

### Windows shortcuts

If you are on Windows, you can use the helper scripts instead of typing the
full commands:

```bat
scripts\start_backend.bat
scripts\start_app.bat
scripts\train_model.bat
```

`start_app.bat` opens the backend and the browser together. `start_backend.bat`
only launches the API, and `train_model.bat` retrains the classifier.

## Optional GitHub API integration

This project can also proxy a symptom-checker API pattern from the GitHub repo
[`priaid-eHealth/symptomchecker`](https://github.com/priaid-eHealth/symptomchecker),
which uses the Infermedica diagnosis flow. To enable the optional backend
endpoint, set these environment variables before starting FastAPI:

Copy `.env.example` to `.env` and fill in the values, or export the variables
directly in your shell.

```bash
export INFERMEDICA_APP_ID="your-app-id"
export INFERMEDICA_APP_KEY="your-app-key"
```

The proxy endpoint is `POST /github/infermedica/diagnosis` and expects the same
`sex`, `age`, and `evidence` structure shown in that GitHub sample.

## Retraining the model

```bash
cd model
pip install -r requirements.txt
python3 train.py
```

On Windows, `scripts\train_model.bat` does the same thing from the project
root.

This re-fits the classifier and overwrites the `.joblib` and `.json` files.
If the local CSVs are missing, `train.py` downloads the Kaggle dataset
`choongqianzheng/disease-and-symptoms-dataset` with `kagglehub` and uses the
CSV files it finds in that download. You can still swap in your own dataset
(same column format: one column per symptom, 0/1 values, last column
`prognosis`) to retrain on different data.

## Deploying so *others* can actually use it

Right now the frontend calls `http://localhost:8420`, which only exists on
your machine. To make this a real web app:

1. **Deploy the backend** somewhere public — Render, Railway, and Fly.io all
   have free tiers that work well for a small FastAPI app. Point them at
   `backend/` with `uvicorn main:app --host 0.0.0.0 --port $PORT` as the start
   command.
2. **Update `API_URL`** near the top of `frontend/index.html`'s `<script>`
   block to your deployed backend's URL.
3. **Host the frontend** — since it's a single static HTML file, Netlify,
   Vercel, GitHub Pages, or Cloudflare Pages all work with zero config.
4. **Lock down CORS** in `backend/main.py` — right now `allow_origins=["*"]`
   is open to any site, which is fine for local testing but should be
   restricted to your actual frontend domain before real use.

## Known limitations (be upfront about these if you share this)

- The training data is synthetic and small — each disease maps to only
  5-10 unique symptom combinations after removing duplicates. The 97%+ test
  accuracy reflects clean, non-overlapping patterns in this specific dataset,
  **not** real-world diagnostic accuracy.
- Real symptoms overlap heavily across conditions and vary by patient history,
  severity, and timing — none of which this model sees.
- This has not been validated by medical professionals and should not inform
  real health decisions. Keep the in-app disclaimer visible if you deploy it,
  and don't remove or downplay it.

## Suggested next steps if you keep building this

- Get a larger, more realistic dataset (with clinician input if possible)
- Add symptom severity/duration as inputs, not just presence/absence
- Report calibrated confidence (Platt scaling / isotonic regression) rather
  than raw Random Forest probabilities
- Add logging + a feedback mechanism so you can see where it's wrong
- Have an actual clinician review the disease list and typical outputs
# medicalassistantproject-Data-engineering-task1-
