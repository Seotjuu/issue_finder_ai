# React Issue Finder AI Backend

FastAPI backend for the React Issue Finder AI demo.

## Project Structure

```text
backend/
  collect_issues.py
  issue_eda.py
  train_models.py
  main.py
  dataset/
    issues.json
  saved_models/
    mlp.pkl
    lstm.keras
    gru.keras
    metrics.json
  training/
    pipeline.py
    predictor.py
```

## Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## REST API

- `GET /api/health`
- `POST /api/analyze`
- `GET /api/clusters`

Example request:

```json
{
  "text": "Maximum update depth exceeded",
  "model": "GRU"
}
```

## Dataset Collection

```bash
set GITHUB_TOKEN=your_personal_access_token
python collect_issues.py
```

The collector saves GitHub Issue data to `dataset/issues.json`.

## EDA

```bash
python issue_eda.py
```

## Training

Train models from `dataset/issues.json`.

```bash
python train_models.py
```

Generated files:

- `saved_models/mlp.pkl`
- `saved_models/lstm.keras`
- `saved_models/gru.keras`
- `saved_models/metrics.json`

## Prediction API

```bash
uvicorn main:app --reload --port 8000
```

```http
POST /predict
Content-Type: application/json
```

```json
{
  "text": "Cannot read properties of undefined"
}
```

```json
{
  "category": "bug",
  "confidence": 0.92
}
```
