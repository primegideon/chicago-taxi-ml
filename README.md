# Throwback on Internship Program

🔭 **What I worked on:**
* **Data Science Internship @ Thrive Africa:** Collaborated on [a ride-share demand analysis project](https://github.com/data-analytics-internship-2026/project-4-team-2).
* **Predictive Modeling:** Engineered time and payment features to build a Random Forest Regressor for exact fare prediction (MAE: $2.78) and a Random Forest Classifier for tip prediction (Accuracy: 94%).

## 🏗️ System Architecture

```mermaid
graph TD;
    A[Raw Chicago Taxi Data] --> B(Pandas Data Engineering);
    B -->|Time/Day Extraction| C{Feature Matrix};
    B -->|Payment Encoding| C;
    C --> D[Random Forest Regressor];
    C --> E[Random Forest Classifier];
    D -->|Joblib Serialization| F(fare_predictor.pkl);
    E -->|Joblib Serialization| G(tip_predictor.pkl);
    F --> H[Streamlit UI];
    G --> H;
    H -->|User Input| I((Real-Time Dashboard));

🌱 **I’m currently learning:**
* Intermediate Python for Data Analysis
* SQL and Git/GitHub workflows
* Pursuing Data Analytics Professional Certificates

🛠️ **Tech Stack & Tools:**

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![SQL](https://img.shields.io/badge/sql-%23336791.svg?style=for-the-badge&logo=sqlite&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) 

📫 **How to reach me:**
* **LinkedIn:** [gideon-amoah09](https://www.linkedin.com/in/gideon-amoah09)
* **Email:** gideon.owusu006@stu.ucc.edu.gh

---
*"Combining actuarial precision with data-driven innovation."*
