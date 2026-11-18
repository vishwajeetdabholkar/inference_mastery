#!/usr/bin/env python3
"""Lab 3.1 — tool-calling & structured-output conformance: "HTTP 200 is not PASS".

Every check sends a real request and verifies the *behaviour*, not just the status code.
This is the method that caught a production bug where parallel_tool_calls=false was
accepted (200) but stripped by a gateway layer before reaching the engine.

Checks:
  single         one obvious tool call, valid JSON arguments
  parallel       3 independent lookups -> expect >1 tool calls (model permitting)
  parallel=false same prompt with parallel_tool_calls=false -> expect <=1 tool call
  choice=none    tool_choice="none" -> expect 0 tool calls
  choice=req     tool_choice="required" on a chit-chat prompt -> expect >=1 tool call
  stream-args    streamed tool-call argument fragments reassemble into valid JSON
  json-schema    response_format json_schema -> output parses and has the required keys

Run it across models and compare: gateways, parsers and chat templates differ per model.

Examples:
  python3 labs/tool_calls.py
  MODEL=moonshotai/Kimi-K3.1-Pro python3 labs/tool_calls.py
"""
import json

from common import MODEL, chat, save_csv

WEATHER = {"type": "function", "function": {
    "name": "get_weather", "description": "Get the current weather for one city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
THREE_CITIES = [{"role": "user", "content": "What's the weather right now in Pune, Tokyo and Lima? Use the tool."}]


def valid_args(tc):
    try:
        return isinstance(json.loads(tc["arguments"] or "{}"), dict)
    except ValueError:
        return False


def check(name, expect, r, passed):
    status = "PASS" if (r["ok"] and passed) else ("ERROR" if not r["ok"] else "FAIL")
    n = len(r.get("tool_calls") or [])
    detail = f"http={r['status']} tool_calls={n}" + (f" error={r.get('error', '')[:90]}" if not r["ok"] else "")
    print(f"  {status:<5} {name:<15} expect: {expect:<38} got: {detail}")
    return {"model": MODEL, "check": name, "result": status, "tool_calls": n, "http": r["status"]}


def main():
    print(f"model={MODEL}")
    rows = []
    r = chat([{"role": "user", "content": "What's the weather in Pune? Use the tool."}],
             tools=[WEATHER], max_tokens=512, stream=False)
    tcs = r.get("tool_calls") or []
    rows.append(check("single", "1 call, valid JSON args", r, len(tcs) == 1 and valid_args(tcs[0])))

    r = chat(THREE_CITIES, tools=[WEATHER], max_tokens=1024, stream=False)
    rows.append(check("parallel", ">1 calls (model may do 1 per turn)", r, len(r.get("tool_calls") or []) > 1))

    r = chat(THREE_CITIES, tools=[WEATHER], parallel_tool_calls=False, max_tokens=1024, stream=False)
    rows.append(check("parallel=false", "<=1 call (is the flag enforced?)", r, len(r.get("tool_calls") or []) <= 1))

    r = chat(THREE_CITIES, tools=[WEATHER], tool_choice="none", max_tokens=512, stream=False)
    rows.append(check("choice=none", "0 calls", r, len(r.get("tool_calls") or []) == 0))

    r = chat([{"role": "user", "content": "Hi! How are you today?"}], tools=[WEATHER],
             tool_choice="required", max_tokens=512, stream=False)
    rows.append(check("choice=req", ">=1 call even on chit-chat", r, len(r.get("tool_calls") or []) >= 1))

    r = chat(THREE_CITIES, tools=[WEATHER], stream=True, max_tokens=1024)
    tcs = r.get("tool_calls") or []
    rows.append(check("stream-args", "fragments -> valid JSON", r, bool(tcs) and all(valid_args(t) for t in tcs)))

    schema = {"type": "object", "properties": {"city": {"type": "string"}, "risk": {"type": "string",
              "enum": ["low", "medium", "high"]}, "reason": {"type": "string"}}, "required": ["city", "risk", "reason"],
              "additionalProperties": False}
    r = chat([{"role": "user", "content": "Assess flood risk for Mumbai in monsoon season."}], max_tokens=1024, stream=False,
             response_format={"type": "json_schema", "json_schema": {"name": "risk", "schema": schema, "strict": True}})
    try:
        obj = json.loads(r.get("text") or "")
        ok = all(k in obj for k in schema["required"]) and obj["risk"] in ("low", "medium", "high")
    except (ValueError, TypeError):
        ok = False
    rows.append(check("json-schema", "parses, required keys, enum respected", r, ok))

    print(f"\n{sum(x['result'] == 'PASS' for x in rows)}/{len(rows)} PASS. A FAIL with http=200 is the interesting kind:")
    print("the API accepted the request and silently did something else. Find which layer dropped it.")
    print(f"saved -> {save_csv('tool_calls', rows)}")


if __name__ == "__main__":
    main()
