"""Lab 2 tools: the agent's actuators.

The model never runs these. It only writes text naming a tool; react_agent.py
parses that text, calls the function here and feeds the result back as an
Observation.
"""
import ast
import operator as op

import requests

# --- Calculator -------------------------------------------------------------
# Only these operators are allowed. Anything else (names, calls, attributes,
# imports) is rejected, so the model can never make us run arbitrary code.
OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul,
       ast.Div: op.truediv, ast.USub: op.neg}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in OPS:
        return OPS[type(node.op)](_eval(node.operand))
    raise ValueError("Unsupported expression")


def calculator(expression: str) -> str:
    """Safely evaluate arithmetic such as '4*185 + 3*95'. Never uses eval()."""
    try:
        # Allow "1,200" and "x" for multiply, which models sometimes write.
        expr = expression.replace(",", "").replace("x", "*").replace("×", "*")
        result = _eval(ast.parse(expr, mode="eval").body)
        # Show 175 rather than 175.0 when the answer is a whole number.
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# --- Wikipedia lookup ---------------------------------------------------------
def wiki_lookup(title: str) -> str:
    """Return the first ~800 characters of a Wikipedia page summary."""
    title = title.strip().strip("\"'")
    url = ("https://en.wikipedia.org/api/rest_v1/page/summary/"
           + title.replace(" ", "_"))
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "DCIT403-lab"})
    except requests.RequestException as e:
        return f"Error: could not reach Wikipedia ({e.__class__.__name__})."
    if r.status_code != 200:
        return f"No Wikipedia page found for '{title}'."
    extract = r.json().get("extract", "")
    return extract[:800] if extract else f"The page '{title}' has no summary."


# Name -> function. The agent looks tools up here, so adding a tool means
# adding one line here and one line in the system prompt.
TOOLS = {
    "calculator": calculator,
    "wiki_lookup": wiki_lookup,
}


if __name__ == "__main__":
    # Step 2 checkpoint: test the tools directly, before any model is involved.
    print("calculator('4*185 + 3*95')   ->", calculator("4*185 + 3*95"))
    print("calculator('1200 - 1025')    ->", calculator("1200 - 1025"))
    print("calculator('1,200 / 4')      ->", calculator("1,200 / 4"))
    print("calculator(\"__import__('os')\") ->", calculator("__import__('os')"))
    print("wiki_lookup('Bono East Region') ->", wiki_lookup("Bono East Region")[:200], "...")
    print("wiki_lookup('Xyzzy Flying Carpet Of Makola') ->",
          wiki_lookup("Xyzzy Flying Carpet Of Makola"))
