

# Nested Argument

If any tool is using the nested object as the argument then while passing the argument, its custom Nested type/class should be clearly mention over there as the type of argument.

# Human In The Loop

The command method takes only 4 type of arguments.
- approve
- edit
- reject
- respond

# For the tools which are going under HITL, if it is expecting the Toolruntime then pass all the necessary attributes while invoking the agent.

If the tool is using store, or config then we need to pass the store or config during the invocation of the agent.