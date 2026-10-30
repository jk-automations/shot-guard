"""Queue an API-format graph that uses paid partner nodes. The Comfy API key is read from a file inside this process,
sent only in the /prompt body (extra_data.api_key_comfy_org, where partner nodes read it) and never printed.
python tools/queue_partner.py graph-api.json [--url http://127.0.0.1:8188] [--key-file ~/.config/jk/comfy_api_key]"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import shotguard_core as sg  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("graph")
ap.add_argument("--url", default="http://127.0.0.1:8188")
ap.add_argument("--key-file", default=os.path.expanduser("~/.config/jk/comfy_api_key"))
a = ap.parse_args()
key = open(a.key_file).read().strip()
r = sg.retry.post_prompt(a.url, json.load(open(a.graph)), client_id="shot-guard-real", extra_data={"api_key_comfy_org": key})
print("queued", r.get("prompt_id"), "errors:", json.dumps(r.get("node_errors") or {})[:300])
