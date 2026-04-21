# MLOps AB Testing API

A FastAPI-based machine learning service that implements A/B testing between two different model versions for Iris flower classification.

## Quick Start

### 1. Install Dependencies
`ash
pip install -r requirements.txt
`

### 2. Run the API
`ash
uvicorn ab_testing_api:app --reload --host 0.0.0.0 --port 8000
`

### 3. Test A/B Testing

**AB Testing Command:**
`ash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"features": [5.1, 3.5, 1.4, 0.2]}'
`

## Browser Testing

Visit **http://localhost:8000/docs** for interactive API documentation and testing.

## API Details

- **Endpoint**: POST /predict
- **A/B Split**: 70% Model A, 30% Model B
- **Models**: model_v1.pkl (A) and model_v3.pkl (B)

## Docker

`ash
docker build -t iris-ab-testing:latest .
docker run -p 8000:8000 iris-ab-testing:latest
`

## CI/CD

Push to main branch to trigger GitHub Actions pipeline.
