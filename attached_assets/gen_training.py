#!/usr/bin/env python3
import os
import json
import argparse
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert story snippets to JSONL for LoRA training"
    )
    parser.add_argument("-i", "--input", default="my_stories.txt")
    parser.add_argument("-o", "--output", default="training_data.jsonl")
    parser.add_argument("--api-url", default="http://127.0.0.1:5000/v1",
                        help="URL of your local LLM API endpoint")
    parser.add_argument("--split-token", default="<SPLIT>")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def build_instruction_prompt(snippet: str) -> str:
    return f"""
Below is a short story snippet. Analyze its tone, conflict, characters, and style, then craft a clear one sentence instruction to generate a new scene in the same spirit. Keep it short and to the point.

Examples:
1) “Write a tense exchange between two rivals in a dimly lit alley, focusing on clipped dialogue and shifting power dynamics.”
2) “Create a reflective morning scene where a character wrestles with a difficult decision, using vivid sensory details and internal monologue.”
3) “Generate an urgent chase through a crowded marketplace, emphasizing fast pacing, narrow escapes, and atmospheric sound cues.”

Now, here is your snippet:
---
{snippet}
---

Write one paragraph that captures the mood, conflict, and stylistic direction for the AI. Keep it focused on creativity and inspiration.
"""

def build_input_prompt(snippet: str) -> str:
    return f"""
Below is a short story snippet. Extract only the essential setup that a writer would need to begin crafting a similar scene: approximate ages and genders of the main characters, the setting (time of day, location, era), and the central situation or conflict. Be as brief as possible.

Here is the snippet:
---
{snippet}
---

Provide that context in a single paragraph with no additional commentary.
"""


def stream_llm_response(prompt: str, args, max_tokens: int) -> str:
    url = f"{args.api_url}/chat/completions"
    headers = {"Content-Type": "application/json"}
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.8,
        "max_tokens": max_tokens,
        "stream": True
    }

    resp = requests.post(url, headers=headers, json=payload, stream=True)
    resp.raise_for_status()

    text = ""
    token_count = 0
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        chunk = line[6:].decode("utf-8").strip()
        if chunk == "[DONE]":
            break
        data = json.loads(chunk)
        delta = data["choices"][0]["delta"].get("content", "")
        if delta:
            text += delta
            token_count += 1
            print(f"Token count: {token_count}", end="\r")

    print()  # newline after final count
    return text.strip()

def process_snippet(snippet: str, idx: int, total: int, args) -> dict:
    logging.info("Processing snippet %d/%d", idx+1, total)
    instr = stream_llm_response(build_instruction_prompt(snippet), args, max_tokens=64)
    inp   = stream_llm_response(build_input_prompt(snippet),   args, max_tokens=64)
    return {
        "instruction": instr,
        "input":      inp,
        "output":     snippet
    }

def main():
    args = parse_args()
    setup_logging()

    if args.overwrite and os.path.exists(args.output):
        os.remove(args.output)

    with open(args.input, encoding="utf-8") as f:
        snippets = [s.strip() for s in f.read().split(args.split_token) if s.strip()]

    total = len(snippets)
    logging.info("Found %d snippets", total)

    with open(args.output, "a", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=args.workers) as pool:

        futures = {
            pool.submit(process_snippet, snip, i, total, args): i
            for i, snip in enumerate(snippets)
        }
        for future in as_completed(futures):
            i = futures[future]
            try:
                record = future.result()
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                logging.info("Snippet %d written", i+1)
            except Exception as e:
                logging.error("Snippet %d failed: %s", i+1, e)

    logging.info("Done. Output → %s", args.output)

if __name__ == "__main__":
    main()
