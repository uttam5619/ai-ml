

# interrupt()
interrupt() pauses the graph execution and sends a value back to the caller.

# Command(resume={})


# Working of interrupt when the control hits the node for the first time vs when the control hits the node for the second time after resuming the execution.
```
Graph execution
      │
      ▼
approvalNode starts
      │
      ▼
interrupt({...})
      │
      ├── PAUSE GRAPH
      │
      ▼
__interrupt__ returned to your application
      │
      ▼
Human sees:
"Do you approve delete_file?"
      │
      │
      │ Human responds
      ▼
Command({ resume: true })
      │
      ▼
Graph resumes
      │
      ▼
interrupt({...})  ← execution reaches this line again
      │
      │ LangGraph recognizes this is a resume
      ▼
interrupt() returns true
      │
      ▼
return { approved: true }
      │
      ▼
Next node
```

# Following is the working of interrupt() and Commnad() methods
```
from langgraph.types import interrupt, Command

async def approval_node(state):
    response = interrupt({
        "message": "What would you like to do?",
        "action": state["actions"]
    })

    if response["type"] == "approve":
        return {
            "approved": True
        }

    if response["type"] == "reject":
        return {
            "approved": False
        }

    if response["type"] == "edit":
        return {
            "action": response["action"]
        }

```

After the interrrupt taking the input from the user.


from langgraph.types import interrupt, Command

def take_user_input():
    user_input = input("Enter your choice (yes/no/edit): ").strip().lower()

    if user_input in ["yes", "true", "approved"]:
        return {
            "type": "approve"
        }

    elif user_input in ["no", "false", "reject"]:
        return {
            "type": "reject"
        }

    elif user_input == "edit":
        action = input("What changes do you want to make? ").strip()

        return {
            "type": "edit",
            "action": action
        }

    else:
        print("Invalid choice.")
        return take_user_input()


Resuming the graph execution using the Command()


```
def resume_execution(config):
    user_command = take_user_input()

    return workflow.invoke(
        Command(resume=user_command),
        config=config
    )
```


# Working of the workflow when invoked for the secodn time.

Now when the execution will resumed the graph will re-execute the node.
```
async def approval_node(state):
    response = interrupt({
        "message": "What would you like to do?",
        "action": state["actions"]
    })

    if response["type"] == "approve":
        return {"approved": True}

    if response["type"] == "reject":
        return {"approved": False}

    if response["type"] == "edit":
        return {
            "action": response["action"]
        }
```

If the type.action == 'edit' then route it to the llm.
```
async def llm_node(state):
    response = await llm.ainvoke(
        f"""
        Work on the following action:

        {state["action"]}
        """
    )

    return {
        "result": response.content
    }
```