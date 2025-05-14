import os
import uvicorn
from dotenv import load_dotenv
from pathlib import Path

env_path = Path(__file__).parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

if __name__ == "__main__":
    uvicorn.run("Model.api:app", host="0.0.0.0", port=8000, reload=True)