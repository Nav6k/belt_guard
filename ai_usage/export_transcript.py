"""Convert Claude Code session logs (.jsonl) into readable markdown.

    python ai_usage/export_transcript.py <session.jsonl> [...]

Keeps the user's messages, the assistant's replies and a one-line summary of each tool call
(which file was written/edited, which command was run). Tool outputs and injected system context are dropped:
they're long and the decisions are visible from the conversation itself.
"""
import json
import re
import sys
from pathlib import Path

OUT = Path(__file__).parent
NOISE = re.compile(r"<(system-reminder|task-notification|ide_selection|ide_opened_file)>.*?</\1>", re.S)


def tool_line(block: dict) -> str:
    name, inp = block.get("name", "?"), block.get("input", {})
    detail = inp.get("file_path") or inp.get("description") or inp.get("command") or inp.get("query") or ""
    detail = str(detail).replace("\n", " ")
    return f"- `{name}` {detail[:160]}"


def text_of(content) -> tuple[list[str], list[str]]:
    texts, tools = [], []
    if isinstance(content, str):
        texts.append(content)
    else:
        for b in content:
            if b.get("type") == "text":
                texts.append(b["text"])
            elif b.get("type") == "tool_use":
                tools.append(tool_line(b))
    return texts, tools


def convert(path: Path) -> str:
    lines = [f"# Claude Code session `{path.stem}`\n"]
    for raw in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(raw)
        if rec.get("type") not in ("user", "assistant") or rec.get("isMeta"):
            continue
        texts, tools = text_of(rec["message"]["content"])
        text = NOISE.sub("", "\n".join(texts)).strip()
        if rec["type"] == "user":
            if text and not text.startswith(("Caveat:", "<command-", "<local-command")):
                lines.append(f"\n---\n\n## 🧑 User\n\n{text}\n")
        else:
            if text:
                lines.append(f"\n### 🤖 Claude\n\n{text}\n")
            if tools:
                lines.append("\n" + "\n".join(tools) + "\n")
    return "\n".join(lines)


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        out = OUT / f"claude_code_session_{p.stem[:8]}.md"
        out.write_text(convert(p), encoding="utf-8")
        print("wrote", out)
