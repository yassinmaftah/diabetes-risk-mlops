# API Reference & Usage Examples

The **Diabetes Risk API** is built using **FastAPI** and provides high-performance inference backed by a model tracked and served through **MLflow Model Registry**.

Interactive Swagger documentation is available at `/docs` and ReDoc at `/redoc`.

---

## Base URLs

- **Docker internal network:** `http://api:8000`
- **Host default mapping:** `http://localhost:8000` (or `http://localhost:8005` if configured via `.env`)

---

## Endpoints

### 1. Health Check

Checks whether the FastAPI service is alive and ready to process requests.

- **Method:** `GET`
- **Path:** `/health`
- **Authentication:** None

#### Example Request

```bash
curl -X GET http://localhost:8000/health
```

#### Example Response (`200 OK`)

```json
{
  "status": "ok for test"
}
```

---

### 2. Predict Diabetes Risk

Accepts patient clinical attributes and returns both the raw model prediction cluster (`0` or `1`) and a human-readable risk category (`"high"` or `"low"`).

- **Method:** `POST`
- **Path:** `/predict`
- **Content-Type:** `application/json`

#### Request Schema (`Patient`)

| Field | Type | Constraints | Description |
|---|---|---|---|
| `Glucose` | float | `> 0, < 400` | Plasma glucose concentration (mg/dL) |
| `BloodPressure` | float | `> 0, < 250` | Diastolic blood pressure (mm Hg) |
| `SkinThickness` | float | `> 0, < 100` | Triceps skin fold thickness (mm) |
| `Insulin` | float | `> 0, < 1000` | 2-Hour serum insulin (mu U/ml) |
| `BMI` | float | `> 0, < 100` | Body mass index ($weight\ in\ kg / (height\ in\ m)^2$) |
| `DiabetesPedigreeFunction` | float | `> 0, < 5` | Diabetes pedigree function genetic risk score |
| `Age` | float | `> 0, < 120` | Patient age (years) |

---

#### Case A: High Risk Patient Profile

Patients with elevated glucose, higher BMI, and elevated insulin typically classify into Cluster 0 (High Risk).

##### Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 180.0,
    "BloodPressure": 85.0,
    "SkinThickness": 35.0,
    "Insulin": 250.0,
    "BMI": 35.5,
    "DiabetesPedigreeFunction": 0.8,
    "Age": 55.0
  }'
```

##### Response (`200 OK`)

```json
{
  "prediction": 0,
  "risk_level": "high"
}
```

---

#### Case B: Low Risk Patient Profile

Patients within healthy physiological thresholds classify into Cluster 1 (Low Risk).

##### Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 85.0,
    "BloodPressure": 65.0,
    "SkinThickness": 20.0,
    "Insulin": 60.0,
    "BMI": 22.0,
    "DiabetesPedigreeFunction": 0.15,
    "Age": 25.0
  }'
```

##### Response (`200 OK`)

```json
{
  "prediction": 1,
  "risk_level": "low"
}
```

---

#### Case C: Input Validation Error

FastAPI validates request payloads using Pydantic. If a metric violates biological range constraints (e.g., `Glucose >= 400`), a descriptive `422 Unprocessable Entity` is returned.

##### Request

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Glucose": 500.0,
    "BloodPressure": 80.0,
    "SkinThickness": 30.0,
    "Insulin": 100.0,
    "BMI": 25.0,
    "DiabetesPedigreeFunction": 0.5,
    "Age": 30.0
  }'
```

##### Response (`422 Unprocessable Entity`)

```json
{
  "detail": [
    {
      "type": "less_than",
      "loc": ["body", "Glucose"],
      "msg": "Input should be less than 400",
      "input": 500.0,
      "ctx": { "lt": 400.0 }
    }
  ]
}
```

---

## Client Code Snippets

### Python (`requests`)

```python
import requests

API_URL = "http://localhost:8000"

payload = {
    "Glucose": 140.0,
    "BloodPressure": 78.0,
    "SkinThickness": 28.0,
    "Insulin": 150.0,
    "BMI": 31.2,
    "DiabetesPedigreeFunction": 0.45,
    "Age": 42.0,
}

response = requests.post(f"{API_URL}/predict", json=payload)
data = response.json()
print(f"Risk Assessment: {data['risk_level'].upper()} (Cluster: {data['prediction']})")
```

### JavaScript / Node.js (`fetch`)

```javascript
const patientData = {
  Glucose: 140.0,
  BloodPressure: 78.0,
  SkinThickness: 28.0,
  Insulin: 150.0,
  BMI: 31.2,
  DiabetesPedigreeFunction: 0.45,
  Age: 42.0,
};

fetch("http://localhost:8000/predict", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(patientData),
})
  .then((res) => res.json())
  .then((data) => console.log("Prediction:", data));
```
