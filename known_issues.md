# Known Issues & Bug Tracking

This document tracks active bugs, resolved engineering issues, and system limitations for the Streamlit ML Dashboard.

## 🔴 Active Issues / Limitations

* **Model File Size Limits (GitHub)**
  * **Description:** The `fare_predictor.pkl` and `tip_predictor.pkl` files are ~54.5 MB. While this is under GitHub's 100 MB hard limit, it triggers a `GH001` Large File Warning during pushes. 
  * **Potential Fix:** If the models grow larger in future iterations, the repository will need to migrate to Git LFS (Large File Storage) to handle the deployment.
* **Localhost Dependency**
  * **Description:** The application currently runs exclusively on a local Streamlit server. 
  * **Next Steps:** Requires containerization (Docker) or cloud hosting (AWS App Runner / Streamlit Community Cloud) for public web access.

## 🟢 Resolved Issues

* **Streamlit FileNotFoundError on `.pkl` Loading**
  * **Description:** During initial Streamlit deployment, the app crashed with `FileNotFoundError: [Errno 2] No such file or directory` when attempting to load the models via Joblib.
  * **Root Cause:** A pathing mismatch caused by VS Code executing the terminal command from the parent directory rather than the specific script directory, causing relative paths (`joblib.load('tip_predictor.pkl')`) to fail.