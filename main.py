"""
Launch the FastAPI server.
Run from the project root with the tripletex_agent conda env active:

    python main.py
"""
import uvicorn

if __name__ == "__main__":
    uvicorn.run("src.app:app", host="0.0.0.0", port=8000, reload=True)
