# Lab 2 evidence

Model: `groq/openai/gpt-oss-20b` (Groq free tier), called through LiteLLM.

| File | What it shows |
|---|---|
| `tools_check.txt` | Step 2: both tools called directly; the calculator rejects `__import__('os')` |
| `baseline.md` | Step 1: the model alone, no tools (3 budget answers, 2 factual answers) |
| `trace_1.txt` | Step 4: the budget question, with the calculator called twice; also an unparseable `Finish:` reply handled by a corrective Observation |
| `trace_2.txt`, `trace_3.txt` | Questions that need both tools |
| `trace_4.txt` | A Wikipedia page that does not exist |
| `trace_5.txt` | A question that needs no tool |
| `trace_6.txt` | A deliberately badly worded question |
| `trace_6_rate_limit.txt` | Same question during a Groq rate limit (HTTP 429): handled without crashing |
| `failure_run1/` | The first run, before the fix: every question failed (see below) |

## Failure and fix

**What happened.** In the first run all six questions stopped at step 1 with
`BadRequestError ... tool_use_failed: Tool choice is none, but model called a tool`.

**Why.** gpt-oss-20b is trained for native function calling. Seeing a list of tools in the
system prompt, it emitted a native tool call instead of the plain-text
`Thought / Action / Action Input` protocol. Our Lab 2 agent sends no tool definitions,
so Groq rejected the call. The model's *decision* (which tool, which input) was fine;
only the *protocol* was wrong.

**Fix.** (1) The prompt now says the agent has no function calling and must reply in plain
text. That alone did not stop it. (2) The loop now reacts to the rejection: Groq's error
contains the rejected call (`failed_generation`), so `recover_native_call()` rebuilds it
as `Thought / Action / Action Input` text and the loop continues normally. Errors with no
usable action get the same corrective Observation as an unparseable reply.

**Result.** All six questions now finish with correct answers (traces 1 to 6).

**Also visible in `trace_3.txt`:** the model ignored the one-step rule and wrote a whole
plan with a *made-up* area (238,535 km²). It never wrote `Observation:`, so the stop
sequence did not cut it off. The parser only acts on the first Action, so the real
Wikipedia observation (239,567 km²) replaced the guess, and the final answer used the
real figure.
