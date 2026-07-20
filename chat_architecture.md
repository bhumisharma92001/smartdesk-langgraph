# "hi" comes to --> api.py (def chat)

1) identities = request.app.state.identities
   (already created at server startup, NOT created here)

2) authorize() --> auth.py (def authorize)
   makes sure browser token hash == user's token hash in `users` table

3) thread() --> auth.py (def thread)
   if thread_id is None      -> create new thread_id
   if thread_id is given      -> check it belongs to this user_id

4) allow() --> request_control.py (def allow)
   checks user hasn't sent more than 10 requests in 60 sec

5) thread_lock() --> request_control.py (def thread_lock)
   locks this thread_id so no 2nd request can run on it at same time

6) graph.invoke(HumanMessage("hi"), user_id) --> api.py
   loads old messages from checkpoint DB (smartdesk_checkpoints.sqlite)
   + appends new "hi" to messages list

7) begin_turn_node() --> graph.py
   clears: error, research_summary, task_output, writer_output
   (does NOT clear: messages, summary, task_queue)

8) supervisor_node() --> graph.py
   _latest_human_message() gets "hi"
   normalize -> "hi"
   "hi" is in _GREETINGS set --> NO LLM call needed here
   sets active_agent = "chat"

9) _route() --> graph.py
   active_agent == "chat" --> goes to chat_node

10) chat_node() --> graph.py
    fast_model().invoke(state["messages"])  <-- THIS is the LLM call
    LLM gets: full messages list only (no user_id, no task_queue, nothing else)
    LLM replies: "Hello! How can I help you?"
    reply appended as AIMessage to messages

11) finalize_node() --> graph.py
    active_agent != "research_task" --> does nothing, just passes through

12) graph reaches END
    new state (messages + patches) saved back to checkpoint DB

13) return ChatResponse(...) --> api.py
    answer = result["messages"][-1].content
    this is the line that actually sends JSON back to frontend

14) frontend receives response --> app.js
    bubble("assistant", data.answer, data.active_agent)
    shows the reply on screen

15) (in background, AFTER response sent)
    _maybe_summarize() --> api.py
    if conversation > 12 human turns -> summarize old messages
    "hi" is turn 1, so this does nothing right now