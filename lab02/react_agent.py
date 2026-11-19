"""Lab 2: a ReAct agent from scratch (no framework).

Loop: the model writes Thought + Action (deliberative), our code runs the tool
and appends an Observation (a new percept), and the model reacts to it on the
next step, until it chooses `finish` or we hit max_steps.

Usage:
    python lab02/react_agent.py                 # run all lab questions, save traces
    python lab02/react_agent.py 3               # run question 3 only
    python lab02/react_agent.py "your question" # run any question (saved as trace_custom.txt)
"""
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
import litellm
from litellm import completion

from tools import TOOLS

load_dotenv()
litellm.suppress_debug_info = True  # hide LiteLLM's "Give Feedback" banner on errors
MODEL = os.getenv("LLM_MODEL")
EVIDENCE = Path(__file__).parent / "evidence"

# --- Step 3: the ReAct prompt --------------------------------------------------
SYSTEM_PROMPT = """You are a helpful assistant for a trader at Makola Market in Accra.
You answer questions about costing (in Ghana cedis, GHS) and quick facts.

You have these tools:
- calculator: evaluates an arithmetic expression using + - * / and brackets. Input: the expression only, e.g. 4*185 + 3*95
- wiki_lookup: returns the summary of an English Wikipedia page. Input: the exact page title, e.g. Bono East Region
- finish: ends the task. Input: your final answer to the trader.

Rules:
- Never do arithmetic in your head. Always use the calculator, even for simple sums.
- Use wiki_lookup for facts. If the page is not found, say you could not find it; do not guess.
- If no tool is needed, go straight to finish.
- Take ONE step at a time and then stop. Never write an Observation yourself; it will be given to you.
- You do NOT have function calling. Never emit a function or tool call. Choose a tool only by
  writing its name as plain text on the Action line, in your normal reply.

Use exactly this format on every step:
Thought: <your reasoning>
Action: <one of: calculator, wiki_lookup, finish>
Action Input: <input for the tool, or the final answer if finish>"""

FORMAT_REMINDER = (
    "Error: I could not find an Action in your reply. Reply in exactly this format:\n"
    "Thought: <your reasoning>\n"
    "Action: <one of: calculator, wiki_lookup, finish>\n"
    "Action Input: <input>"
)

# --- Step 3: the parser ---------------------------------------------------------
# Tolerant of extra spaces, markdown bold (**Action:**) and different case.
ACTION_RE = re.compile(
    r"\**Action\**\s*:\**\s*([\w-]+)\**\s*\n+\s*\**Action Input\**\s*:\**\s*(.*)",
    re.DOTALL | re.IGNORECASE,
)


def parse(text):
    """Return (action, action_input) from a model reply, or (None, None)."""
    m = ACTION_RE.search(text or "")
    if not m:
        return None, None
    action = m.group(1).strip().lower()
    action_input = m.group(2).strip()
    # If the model ignored the stop sequence and kept going, cut at the next step.
    action_input = re.split(r"\n\s*(?:Observation|Thought)\s*:", action_input)[0].strip()
    if action != "finish":
        # Tool inputs are one line; drop stray backticks or quotes around them.
        action_input = action_input.splitlines()[0].strip().strip("`")
    return action, action_input


# --- The model call -------------------------------------------------------------
def call_model(messages):
    """One think step. stop=["Observation:"] stops the model inventing percepts."""
    resp = completion(
        model=MODEL,
        messages=messages,
        temperature=0,
        stop=["Observation:"],
    )
    text = resp.choices[0].message.content or ""
    tokens = resp.usage.total_tokens if resp.usage else 0
    return text.strip(), tokens


def recover_native_call(error):
    """If Groq rejected a native tool call, rebuild it as Thought/Action/Action Input text.

    Groq's error includes the rejected call as "failed_generation", e.g.
    {"name": "calculator", "arguments": {"input": "1200-1025"}}. Returns None if
    there is nothing usable to recover.
    """
    msg = str(error)
    if "tool_use_failed" not in msg:
        return None
    try:
        body = json.loads(msg[msg.index("{"):])          # Groq's JSON error body
        call = json.loads(body["error"]["failed_generation"])
        name = call["name"]
        args = call.get("arguments") or {}
        if isinstance(args, str):
            args = json.loads(args)
    except (ValueError, KeyError, TypeError):
        return None
    # The model sometimes nests our own format inside the arguments.
    name = args.get("Action", name)
    value = args.get("Action Input")
    if value is None:
        value = next((v for k, v in args.items() if k not in ("id", "Thought")), "")
    thought = args.get("Thought", "(the model chose this action as a native tool call)")
    return f"Thought: {thought}\nAction: {name}\nAction Input: {value}"


# --- Step 4 and 5: the loop -------------------------------------------------------
def run_agent(question, max_steps=6, verbose=True):
    """Run the ReAct loop. Returns (final_answer, trace_text, total_tokens)."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}\n\n"
                                    "Reply in plain text starting with 'Thought:'. Do not call a function."},
    ]
    trace = [f"Question: {question}", f"Model: {MODEL}", ""]
    total_tokens = 0

    def log(line):
        trace.append(line)
        if verbose:
            print(line)

    for step in range(1, max_steps + 1):
        log(f"--- Step {step} ---")

        # THINK: ask the model for the next Thought + Action.
        try:
            reply, tokens = call_model(messages)
        except Exception as e:
            # Failure found in the first run: gpt-oss tries to make a *native* tool
            # call, and Groq rejects it because we send no tools. The rejected call
            # is still a valid decision, so translate it into our text protocol.
            recovered = recover_native_call(e)
            if recovered:
                log("[model made a native tool call; translated to the text protocol]")
                reply, tokens = recovered, 0
            elif "output_parse_failed" in str(e) or "tool_use_failed" in str(e):
                # No usable action at all: treat like an unparseable reply.
                log("[model gave no usable text; asking for the plain-text format]")
                messages.append({"role": "user", "content": "Observation: " + FORMAT_REMINDER})
                continue
            else:
                # Anything else: rate limit (429), network, bad model name...
                log(f"[model error] {e.__class__.__name__}: {e}")
                answer = "Sorry, the model could not be reached. Please try again later."
                log(f"Final answer: {answer}")
                return answer, "\n".join(trace), total_tokens
        total_tokens += tokens
        log(reply)
        messages.append({"role": "assistant", "content": reply})

        # PARSE: turn text into an action. Unparseable -> corrective observation.
        action, action_input = parse(reply)
        if action is None:
            observation = FORMAT_REMINDER
        elif action == "finish":
            log(f"\nFinal answer: {action_input}")
            log(f"Total tokens: {total_tokens}")
            return action_input, "\n".join(trace), total_tokens
        elif action not in TOOLS:
            observation = (f"Error: unknown action '{action}'. "
                           f"Use one of: {', '.join(TOOLS)}, finish.")
        else:
            # ACT: our code runs the tool, never the model.
            try:
                observation = TOOLS[action](action_input)
            except Exception as e:
                observation = f"Error: the tool {action} failed ({e})."

        # SENSE: the result goes back as a new percept.
        log(f"Observation: {observation}")
        messages.append({"role": "user", "content": f"Observation: {observation}"})

    answer = f"Stopped: no final answer within {max_steps} steps."
    log(f"\n{answer}")
    log(f"Total tokens: {total_tokens}")
    return answer, "\n".join(trace), total_tokens


# --- The lab questions (Step 4 and Step 5) ---------------------------------------
QUESTIONS = [
    # 1. Step 4: the budget question from Part A
    "I have GHS 1,200. I want 4 bags of rice at GHS 185 each and 3 gallons of oil "
    "at GHS 95 each. How much will I have left?",
    # 2. Needs both tools
    "What is the capital of the Bono East Region? If I send 12 bags of yam there "
    "at GHS 240 each, what is the total cost?",
    # 3. Needs both tools
    "Using Wikipedia, find Ghana's area in square kilometres. If it were shared "
    "equally among Ghana's 16 regions, how many square kilometres would each get?",
    # 4. Wikipedia page that does not exist
    "What does Wikipedia say about the Makola Market Flying Carpet Traders Association?",
    # 5. Needs no tool
    "Good morning! Can you greet me politely? I am a new trader here.",
    # 6. Deliberately badly worded
    "rice 4 185 oil 3 95 i got 1200 wat remain??",
]


def save_trace(name, trace):
    EVIDENCE.mkdir(exist_ok=True)
    path = EVIDENCE / f"trace_{name}.txt"
    path.write_text(trace + "\n", encoding="utf-8")
    print(f"[saved lab02/evidence/{path.name}]\n")


def main(args):
    if not MODEL:
        print("LLM_MODEL is not set. Copy .env.example to .env and set it.")
        sys.exit(1)

    if args and not args[0].isdigit():
        _, trace, _ = run_agent(" ".join(args))
        save_trace("custom", trace)
        return

    numbers = [int(args[0])] if args else range(1, len(QUESTIONS) + 1)
    for n in numbers:
        print(f"\n========== Question {n} ==========")
        _, trace, _ = run_agent(QUESTIONS[n - 1])
        save_trace(n, trace)


if __name__ == "__main__":
    main(sys.argv[1:])
