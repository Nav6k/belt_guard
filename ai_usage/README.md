# AI usage

**Tool:** Claude Code (VS Code extension), used throughout the project.

## Files

| File | What it is |
|---|---|
| [claude_code_session_184beee2.md](claude_code_session_184beee2.md) | The main build session: planning, all five components, debugging, README |
| [claude_code_session_90b4b7b6.md](claude_code_session_90b4b7b6.md) | A side session where I asked for the project to be explained simply |
| [export_transcript.py](export_transcript.py) | Converts the raw `.jsonl` session logs into these markdown files |

**What the exports contain:**
- my messages;
- the assistant's replies;
- a one-line note of each tool call (the file edited or the command run).

**What was left out:**
- tool outputs;
- injected system context.

They are long and add nothing to the decisions.

One note on the main log. Partway through, the session hit its context limit and was summarised automatically; the "This session is being continued…" message marks that point. The summary lists every decision and bug fix made before it.

## The bigger decisions, and where to find them

1. **Breaking the problem down.** I gave it the brief and asked for a plan first ([PROJECT_PLAN.md](../PROJECT_PLAN.md)), before any code. The plan fixed:
   - the five components;
   - what to skip (LLM agent, message broker, cloud);
   - the order to build in, so that every stage could be run and checked before the next.
2. **Understanding, not just accepting.** I asked for every change to be explained. That became [docs/CODE_EXPLAINED.md](../docs/CODE_EXPLAINED.md).
3. **Choosing the vision dataset.** The assistant searched and listed the options: BeltCrack needed a registration request, Kaggle had mostly concrete cracks, and Roboflow had real belts. I chose Roboflow. The export turned out to have 48 images, and the README reports that honestly instead of hiding it.
4. **Overriding wrong results.** Several first versions were wrong and were caught by running the system and asking whether the output made physical sense. Examples:
   - phantom stuck sensors;
   - shutdown false alarms;
   - a repaired bearing flagged as dangerous;
   - the loop tracker's position-error reasoning.

   See README section 11 for the table, and search the main log for "stuck", "shutdown", "cooler" and "splice".
