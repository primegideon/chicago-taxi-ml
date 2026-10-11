"""
Application Entry Point

Delegates execution to the Streamlit analytics application located in .streamlit/app.py.
Enables launching the interface directly via:
    streamlit run app.py
"""
from pathlib import Path
import runpy

app_path = Path(__file__).resolve().parent / ".streamlit" / "app.py"
runpy.run_path(str(app_path), run_name="__main__")
