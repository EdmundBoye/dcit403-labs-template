# Lab 2: Architectures: A ReAct Agent From Scratch: Concept Check

Answer in your own words before the Wednesday session (independent work). Your answers feed Section 2 of your lab report (Concept and classical root).

## Q1. Classify a thermostat, a chess engine and a ReAct agent as reactive, deliberative or hybrid, with one reason each.

_Your answer:_

- **Thermostat: reactive.** It maps the current percept (the room temperature) straight to an action with a condition-action rule ("if below the set point, heat; if above, stop"). It keeps no model of the world, makes no plan and does not remember past readings.
- **Chess engine: deliberative.** It keeps an internal model of the board and searches ahead through possible move sequences (for example, minimax with alpha-beta pruning) to choose the move that best serves its goal of winning. It plans before it acts.
- **ReAct agent: hybrid.** The *Thought* step is deliberative: the model reasons about the goal and states a plan that spans several steps. Each step is also reactive: the next Thought and Action respond directly to the latest Observation, and the code around the model reacts to errors (unknown tool, unparseable reply, step limit) with fixed rules. It combines a planning layer with a quick response layer.

## Q2. In the trace above, which lines are percepts, and which are actions?

_Your answer:_

- **Percepts** (what the agent receives from its environment):
  - The user's question ("I have GHS 1,200 ... How much will I have left?"), the first percept.
  - `Observation: 1025`, the result of the first calculator call.
  - `Observation: 175`, the result of the second calculator call.
- **Actions** (what the agent does to its environment):
  - `Action: calculator` / `Action Input: 4*185 + 3*95`
  - `Action: calculator` / `Action Input: 1200 - 1025`
  - `Action: finish` / `Action Input: You will have GHS 175 left.` (the final reply to the user)
- The `Thought:` lines are neither. They are the agent's internal deliberation, the part of the agent function that decides which action to take next.

## Q3. Why do we pass `stop=["Observation:"]` when calling the model?

_Your answer:_

The model only produces text. If it is not stopped, it continues the pattern and writes its own `Observation:` line, making up a tool result (for example, a wrong total) and possibly the rest of the trace, without any tool having run. The stop sequence ends generation at the point where an observation would start. Control then goes back to my code, which parses the Action, runs the real tool and adds the real `Observation:` to the conversation. This keeps the percepts true (they come from the environment, not from the model's imagination). It also limits each model call to exactly one Thought and Action, which keeps the parser simple and saves tokens.

## Q4. Map the trace above to the BDI terms belief, desire and intention.

_Your answer:_

- **Beliefs** (what the agent holds true): the facts in the question (budget GHS 1,200; 4 bags of rice at GHS 185; 3 gallons of oil at GHS 95), then the facts added by observations: the total cost is 1025, and the amount left is 175. In the code, the beliefs are the conversation history (messages list), which grows with every observation.
- **Desire** (the goal it would like to achieve): to tell the trader correctly how much money will be left after the purchase.
- **Intentions** (the goal and plan it commits to): the plan stated in the Thoughts: "I need the total cost first", then "Now subtract the cost from the budget". Each `Action: calculator` executes one step of that plan. When the belief "175 left" is in place, the intention is achieved, so the agent commits to `finish`.
- This follows the BDI cycle: perceive (Observation), update beliefs (append to history), choose or keep the intention (Thought), execute one plan step (Action), repeat.
