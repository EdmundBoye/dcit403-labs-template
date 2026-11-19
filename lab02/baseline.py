"""Lab 2, Step 1: the model on its own, with no tools.

Asks the budget question three times and two factual questions, then writes
lab02/evidence/baseline.md. The budget answers are auto-marked (correct = 175);
mark the factual answers yourself by checking them on Wikipedia.

Usage: python lab02/baseline.py
"""
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from litellm import completion

load_dotenv()
MODEL = os.getenv("LLM_MODEL")

BUDGET_Q = ("I have GHS 1,200. I want 4 bags of rice at GHS 185 each and 3 gallons "
            "of oil at GHS 95 each. How much will I have left?")
FACT_QS = [
    "What is the capital of the Bono East Region of Ghana?",
    "In what year was Makola Market in Accra first established?",
]


def ask(question):
    resp = completion(model=MODEL, temperature=1,  # default-like sampling, as a user would see
                      messages=[{"role": "user", "content": question}])
    return (resp.choices[0].message.content or "").strip()


def one_line(text):
    return " ".join(text.split()).replace("|", "/")


def main():
    if not MODEL:
        sys.exit("LLM_MODEL is not set. Copy .env.example to .env and set it.")

    rows = ["# Lab 2 Step 1: Baseline (model alone, no tools)", "",
            f"Model: `{MODEL}`", "",
            "## Budget question (correct answer: GHS 175)", "",
            f"> {BUDGET_Q}", "",
            "| Run | Answer | Correct? |", "|---|---|---|"]
    for i in range(1, 4):
        answer = ask(BUDGET_Q)
        correct = "175" in answer.replace(",", "")
        print(f"Budget run {i}: {answer}\n")
        rows.append(f"| {i} | {one_line(answer)} | {'Yes' if correct else 'No (check)'} |")

    rows += ["", "## Factual questions (check each on Wikipedia and fill in)", "",
             "| Question | Answer | How confident it sounded | Correct? |",
             "|---|---|---|---|"]
    for q in FACT_QS:
        answer = ask(q)
        print(f"{q}\n{answer}\n")
        rows.append(f"| {q} | {one_line(answer)} | _fill in_ | _fill in_ |")

    out = Path(__file__).parent / "evidence" / "baseline.md"
    out.parent.mkdir(exist_ok=True)
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
