## Runtime does have following attributes
context
store
stream_writer
execution_info
server_info





# If we pass the context or store or config directly inside the agent (create_agent) where agent is not a node itself,then agent can not directly utilize it. It is the middleware or tools mentioned with the agent who utilises the context, store or config via runtime.

@before_model middleware -> It takes 2 arguments one is `state` and other is `runtime`
