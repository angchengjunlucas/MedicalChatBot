from __future__ import annotations

import json
import os
import sys

import httpx


API_URL = "http://localhost:8000/chat"


def main() -> None:
    print("Type your question. Press Enter on an empty line to exit.")
    while True:
        try:
            q = input("Question> ").strip()
        except EOFError:
            break
        if not q:
            break

        payload = {
            "user_id": "u1",
            "query": q,
            "history": [],
            "user_metadata": {},
        }
        try:
            resp = httpx.post(API_URL, json=payload, timeout=600.0)
            resp.raise_for_status()
            data = resp.json()
            print("\n--- Response ---")
            print(data.get("response_text", ""))
            detail = data.get("response_detail")
            if detail and os.getenv("CHAT_CLI_SHOW_DETAILS", "").lower() in {"1", "true", "yes"}:
                print("\n--- Details ---")
                print(detail)
            print("--- End ---\n")
        except Exception as exc:
            print(f"Request failed: {exc}")
            try:
                print(resp.text)  # type: ignore[name-defined]
            except Exception:
                pass


if __name__ == "__main__":
    main()
