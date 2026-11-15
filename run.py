"""Start DocuChat:  python run.py  [--port 8000] [--host 127.0.0.1] [--reload]"""
import argparse
import os
from pathlib import Path

import uvicorn

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--llm", choices=["anthropic", "openai_compat", "extractive"],
                        help="override LLM_PROVIDER (extractive needs no API key)")
    args = parser.parse_args()
    if args.llm:
        os.environ["LLM_PROVIDER"] = args.llm

    os.chdir(Path(__file__).parent)  # so .env and data/ resolve next to this file
    uvicorn.run("app.main:create_app", factory=True, host=args.host, port=args.port, reload=args.reload)
