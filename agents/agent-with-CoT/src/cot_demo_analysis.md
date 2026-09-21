# Chain-of-Thought in LangGraph — Analysis of `demo.ipynb`

This document lists what went wrong in your `demo.ipynb` CoT + LangGraph implementation, why it breaks (or weakens) the workflow, and the correct patterns to use instead.

---

## What you were trying to build

A LangGraph workflow where:

1. A user asks a question.
2. The graph runs **chain-of-thought** reasoning in explicit steps.
3. Each step uses context from previous steps.
4. The workflow returns a final answer.

Your overall idea (separate setup → multi-step reasoning → compiled graph) is sound. The problems are mostly in **state updates**, **message handling**, and **how user input flows into the graph**.

---

## Critical bugs (workflow fails or does not reason correctly)

### 1. `reason_and_respond` does not return state updates

**Wrong (`demo.ipynb`):**

```python
def reason_and_respond(state: State):
    for step in steps:
        execute_individiual_step(step, state)
```

**Why it's wrong:** LangGraph nodes must return a dictionary of state updates (e.g. `{"messages": [...]}`). Returning `None` means nothing is written to state after this node. Even if the LLM runs inside the loop, the graph state never gets the step outputs.

**Correct:**

```python
def reason_and_respond(state: State):
    messages = list(state.messages)
    new_messages = []

    for step in steps:
        step_message = HumanMessage(content=f"Now perform the following step: {step}")
        messages.append(step_message)
        new_messages.append(step_message)

        result = llm.invoke(messages)
        messages.append(result)
        new_messages.append(result)

    return {"messages": new_messages}
```

Or accumulate inside the loop and return once at the end.

---

### 2. Step results are never merged back into `state`

**Wrong:**

```python
def reason_and_respond(state: State):
    for step in steps:
        execute_individiual_step(step, state)  # return value ignored
```

Inside `execute_individiual_step`, you build `messages` from `state.messages.copy()`, but `state` is never updated between loop iterations. So every step starts from the **same** message list (only what `ignition` added), not from prior step answers.

**Why it's wrong:** Step 2 and step 3 do not see step 1's LLM output in the conversation history passed to the model (unless you mutate `state` manually, which you do not).

**Correct:** Either update a local `messages` list across the loop (as above), or return from each step and let LangGraph merge via `add_messages`:

```python
def execute_step(step: str, state: State):
    step_message = HumanMessage(content=f"Now perform the following step: {step}")
    result = llm.invoke(state.messages + [step_message])
    return {"messages": [step_message, result]}
```

With separate nodes or a loop node that reads **current** `state.messages` each time.

---

### 3. Returning the full message list with `add_messages` can duplicate history

**Wrong pattern in `ignition` and `execute_individiual_step`:**

```python
messages = state.messages.copy()
messages.append(SystemMessage(...))
messages.append(HumanMessage(...))
return {"messages": messages}  # entire list, including old messages
```

**Why it's wrong:** `messages` uses `Annotated[list[BaseMessage], add_messages]`. The reducer **appends** returned messages to existing state. If you return the full copied list, you re-append messages that are already in state → duplicates and confused context.

**Correct:** Return only **new** messages:

```python
def ignition(state: State):
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Problem:\n{state.question}"),
        ]
    }
```

---

### 4. User question is hardcoded — not driven by user input

**Wrong:**

```python
question = """A database table has 5 million rows..."""

def ignition(state: State):
    messages.append(HumanMessage(content=f"Problem:\n{question}"))
```

**Why it's wrong:** The workflow cannot answer arbitrary user queries. CoT is wired to one static problem.

**Correct:** Put the question in state and pass it at invoke time:

```python
class State(BaseModel):
    question: str = ""
    messages: Annotated[list[BaseMessage], add_messages]

initial_state = {
    "question": "What is the overall performance impact if ...?",
    "messages": [],
}
result = workflow.invoke(initial_state, config=config)
```

Or pass the question only as the first `HumanMessage` in `messages` when invoking.

---

### 5. Dict key typo (seen in `cot.ipynb` run — worth avoiding)

**Wrong:**

```python
return {messages: messages}  # invalid — uses list as dict key
```

**Error:** `TypeError: cannot use 'list' as a dict key (unhashable type: 'list')`

**Correct:**

```python
return {"messages": messages}
```

`demo.ipynb` uses the quoted form correctly; keep that consistent everywhere.

---

## Design issues (works partially but is not proper CoT + LangGraph)

### 6. All CoT steps hidden inside one node

**Your approach:** One node `reason_and_respond` runs a Python `for` loop over `steps`.

**Issue:** You lose LangGraph benefits: per-step visibility, streaming, checkpointing per step, conditional routing, retries on one step, etc.

**Better patterns:**

| Pattern | When to use |
|--------|-------------|
| **One node, internal loop** | Simple demos; fix return/state handling as above |
| **One node per step** | Fixed CoT template (understand → decompose → answer) |
| **Loop with `current_step` in state** | Dynamic number of steps or replanning |

Example — separate nodes (fixed 3-step CoT):

```text
START → ignition → step_understand → step_decompose → step_answer → END
```

Example — loop with step index:

```python
class State(BaseModel):
    question: str
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: int = 0

def route(state: State):
    if state.current_step < len(steps):
        return "execute_step"
    return END
```

---

### 7. `ignition` name and role are unclear

**Current:** `ignition` adds system prompt + problem.

**Better:** Name by responsibility, e.g. `setup_context` or `initialize`, and only add messages once.

---

### 8. Weak CoT prompting

**Current system prompt** is generic (“solve step by step”) and does not require visible reasoning structure.

**Stronger CoT prompt:**

```text
You are an expert problem solver using chain-of-thought reasoning.

For each step:
1. Label the step you are performing.
2. Show your reasoning explicitly before any conclusion.
3. Build on outputs from previous steps.
4. On the final step, give a concise answer after the reasoning.
```

You can also ask the model to use a fixed format (`Reasoning: ...`, `Conclusion: ...`) for easier parsing.

---

### 9. No dedicated `final_answer` field

**Current:** Final answer is only the last `AIMessage` in a long thread.

**Better for production:**

```python
class State(BaseModel):
    question: str = ""
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str = ""
```

Set `final_answer` in the last step node so callers do not have to scrape message history.

---

### 10. Unused imports and naming

- Unused: `ChatPromptTemplate`, `AIMessage` (unless you use them for structured prompts).
- Typo: `execute_individiual_step` → `execute_individual_step`.
- Inconsistent step labels: `step1`, `step2`, `Step3`.

---

## What you did correctly

1. **StateGraph + `State` with Pydantic** — good foundation.
2. **`add_messages` reducer** — correct choice for conversation state.
3. **`InMemorySaver` checkpointer** — fine for local demos and `thread_id` sessions.
4. **Explicit step list** — matches structured CoT (understand → decompose → answer).
5. **Separate LLM call per step** — aligns with sequential CoT (once state is fixed).
6. **Temperature 0** — reasonable for deterministic reasoning demos.

---

## Recommended correct workflow (minimal fix)

This keeps your structure (ignition → multi-step reasoning) but fixes state and user input.

```python
from typing import Annotated
from pydantic import BaseModel
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver

steps = [
    "Step 1: Understand the problem and state what must be solved.",
    "Step 2: Break the problem into sub-problems and solve each sequentially.",
    "Step 3: State the final concise answer.",
]

system_prompt = """You are an expert problem solver using chain-of-thought reasoning.
Follow each step instruction. Use prior step outputs. Do not skip steps."""

class State(BaseModel):
    question: str = ""
    messages: Annotated[list[BaseMessage], add_messages]
    final_answer: str = ""

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0, api_key=openai_key)

def ignition(state: State):
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Problem:\n{state.question}"),
        ]
    }

def reason_and_respond(state: State):
    messages = list(state.messages)
    new_messages = []

    for step in steps:
        step_msg = HumanMessage(content=f"Now perform the following step: {step}")
        messages.append(step_msg)
        new_messages.append(step_msg)

        ai_msg = llm.invoke(messages)
        messages.append(ai_msg)
        new_messages.append(ai_msg)

    return {
        "messages": new_messages,
        "final_answer": ai_msg.content,
    }

graph = StateGraph(State)
graph.add_node("ignition", ignition)
graph.add_node("reason_and_respond", reason_and_respond)
graph.add_edge(START, "ignition")
graph.add_edge("ignition", "reason_and_respond")
graph.add_edge("reason_and_respond", END)

workflow = graph.compile(checkpointer=InMemorySaver())

config = {"configurable": {"thread_id": "cot-demo-1"}}
result = workflow.invoke(
    {
        "question": "A database table has 5 million rows. Adding an index makes SELECT 10x faster but INSERT 20% slower. With 90% SELECTs and 10% INSERTs, what is the overall performance impact?",
        "messages": [],
    },
    config=config,
)
print(result["final_answer"])
```

---

## Recommended correct workflow (LangGraph-native loop)

Better when you want each step as a graph transition (streaming, checkpoints per step):

```python
class State(BaseModel):
    question: str = ""
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: int = 0
    final_answer: str = ""

def setup(state: State):
    return {
        "messages": [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Problem:\n{state.question}"),
        ],
        "current_step": 0,
    }

def execute_one_step(state: State):
    step = steps[state.current_step]
    step_msg = HumanMessage(content=f"Now perform: {step}")
    ai_msg = llm.invoke(state.messages + [step_msg])
    update = {
        "messages": [step_msg, ai_msg],
        "current_step": state.current_step + 1,
    }
    if state.current_step + 1 == len(steps):
        update["final_answer"] = ai_msg.content
    return update

def should_continue(state: State):
    if state.current_step < len(steps):
        return "execute_one_step"
    return END

graph = StateGraph(State)
graph.add_node("setup", setup)
graph.add_node("execute_one_step", execute_one_step)
graph.add_edge(START, "setup")
graph.add_edge("setup", "execute_one_step")
graph.add_conditional_edges("execute_one_step", should_continue)
workflow = graph.compile(checkpointer=InMemorySaver())
```

---

## Quick checklist before running

| Check | Pass? |
|-------|-------|
| Every node returns a dict of state keys | |
| With `add_messages`, return only **new** messages | |
| User question comes from invoke input / state, not a global variable | |
| Each CoT step sees prior LLM outputs in `messages` | |
| `reason_and_respond` (or loop) returns updates to LangGraph | |
| Optional: `final_answer` extracted for the caller | |

---

## Summary

| Area | Your `demo.ipynb` | Correct approach |
|------|-------------------|------------------|
| Node return value | `reason_and_respond` returns nothing | Always `return {"messages": [...], ...}` |
| Step chaining | `state` never updated in loop | Accumulate `messages` locally or use graph loop |
| `add_messages` usage | Return full copied list | Return only new messages |
| User input | Hardcoded `question` variable | `question` in state or initial `HumanMessage` |
| Graph design | All steps in one Python loop | Loop in one node (fixed) or conditional edges (native) |
| Output | Buried in message history | Optional `final_answer` field |

Fixing items 1–4 is enough to get a working CoT LangGraph demo. Items 6–9 improve clarity, observability, and reuse for real user queries.
