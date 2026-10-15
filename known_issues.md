# Engineering Issue Tracking & System Constraints

This document records active system constraints, operational considerations, and resolved engineering defects for the Chicago Taxi machine learning serving infrastructure.

---

## Active System Constraints & Mitigation Strategies

### 1. Serialized Model Artifact File Sizes (GitHub VCS Limits)
* **Status:** Monitored
* **Component:** `models/fare_predictor.pkl`, `models/tip_predictor.pkl`
* **Observation:** Serialized ensemble models are approximately 54.5 MB each. While below GitHub's absolute 100 MB hard limit, commits trigger `GH001` advisory warnings during remote pushes.
* **Mitigation Strategy:** Should subsequent model iterations or ensemble expansion exceed 50 MB, the repository will transition binary asset management to Git LFS (Large File Storage) or an external object store / model registry (e.g., Google Cloud Storage, Amazon S3, or MLflow).

### 2. Runtime Environment & Cloud Deployment Architecture
* **Status:** Planned Enhancement
* **Component:** `app.py`
* **Observation:** The serving interface is currently configured for local and edge execution via Streamlit CLI (`streamlit run app.py`).
* **Deployment Roadmap:** Future milestones encompass containerization using standard OCI/Docker images and automated CI/CD deployment to managed serverless container platforms (e.g., Google Cloud Run, AWS App Runner, or Streamlit Community Cloud).

---

## Resolved Engineering Incidents

### Incident 001: Deserialization Path Resolution Failure (`FileNotFoundError`)
* **Severity:** High (Application Startup Blocker)
* **Component:** Model Ingestion Pipeline (`load_models`)
* **Symptom:** The Streamlit application terminated with `FileNotFoundError: [Errno 2] No such file or directory` during model deserialization via `joblib.load()`.
* **Root Cause Analysis (RCA):** Execution from arbitrary directories caused relative file lookups (`joblib.load('tip_predictor.pkl')`) to resolve against the process working directory rather than the package models directory containing the serialized artifacts.
* **Remediation:** Relocated models to a dedicated `models/` directory, refactored path resolution using `Path(__file__).resolve().parent / 'models'`, and consolidated the main application entry point directly into `app.py`.