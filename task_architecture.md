# "Create a 3-step project launch task" comes to --> api.py (def chat)

1) identities = request.app.state.identities --> api.py:164
   (already created at server startup, NOT created inside chat())

2) authorize() --> auth.py:56-62 (def authorize)
   makes sure browser token hash == user's token hash in `users` table

3) thread() --> auth.py:64-77 (def thread)
   if thread_id is None       -> creates a new thread_id
   if thread_id is supplied   -> checks that it belongs to this user_id

4) allow() --> request_control.py:19-29 (def allow)
   checks user has not sent more than 10 requests in 60 seconds

5) thread_lock() --> request_control.py:31-33 (def thread_lock)
   locks this thread_id so a second request cannot update the same conversation
   while the first request is still running

6) graph.invoke(HumanMessage(user request), user_id) --> api.py:179-182
   loads old GlobalState from `smartdesk_checkpoints.sqlite` using thread_id
   + appends the new HumanMessage to state["messages"]

7) begin_turn_node() --> graph.py:168-175
   always clears: error, research_summary, task_output, writer_output
   does NOT clear: messages, summary, task_queue, completed_tasks, routing_log

8) supervisor_node() --> graph.py:127-165
   _latest_human_message() gets: "Create a 3-step project launch task"
   this is not a greeting and not a short active-task continuation
   router fast_model LLM is called with ROUTER_SYSTEM_PROMPT + latest user message
   expected structured result: Route(workflow="task")
   sets active_agent = "task"

   LLM CALL #1: supervisor routing model

9) _route() --> graph.py:285-287
   active_agent == "task" --> graph goes to task_agent_node
   edge mapping is defined at graph.py:334-347

10) task_agent_node(state, store) --> agents/task_agent.py:75-98
    receives:
      state = current checkpointed GlobalState
      store = persistent `smartdesk_memory.sqlite` store injected by LangGraph

11) _build_task_agent(store) --> agents/task_agent.py:26-36
    builds the internal TaskAgent once and reuses it from lru_cache
    TaskAgent contains:
      reasoning_model()
      TASK_SYSTEM_PROMPT
      create_task tool
      get_task tool
      mark_step_done tool
      save_note tool
      TaskState containing user_id

12) context is built --> agents/task_agent.py:78-81

    context = [
        *build_conversation_summary_context(state),
        *_task_context(state),
    ]

    build_conversation_summary_context(state):
      if an older conversation summary exists -> adds it as SystemMessage
      otherwise -> returns []

    _task_context(state) --> agents/task_agent.py:61-72:
      if task_queue contains active tasks -> adds their exact JSON as SystemMessage
      if task_queue is empty -> returns []

    For a first new-task request, normally:
      summary = ""
      task_queue = []
      context = []

    IMPORTANT: context is only a local Python list here.
    It is passed into run_agent() and added to the TaskAgent's input messages there.

13) run_agent(...) --> agents/common.py:110-174
    called from agents/task_agent.py:82-85 with:
      agent = TaskAgent
      state = current GlobalState
      context = summary context + active-task context
      max_retries = 1
      keep_last = 10

14) compact_history() --> agents/common.py:83-107
    reads state["messages"]
    removes old ToolMessages, SystemMessages and AI tool-call messages
    retains the latest 10 normal HumanMessage/AIMessage conversation messages

15) input_messages are assembled --> agents/common.py:139-142

    input_messages = context + compact_history(state["messages"])

    New-task example normally becomes:
      [HumanMessage("Create a 3-step project launch task")]

    Existing-task example may become:
      [SystemMessage(active task JSON), ..., HumanMessage("done")]

16) agent.invoke(...) --> agents/common.py:149-152

    agent.invoke({
        "messages": input_messages,
        "user_id": state["user_id"]
    })

    The TaskAgent model receives conceptually:
      TASK_SYSTEM_PROMPT
      available tool names + Pydantic tool schemas
      earlier summary, if present
      active task JSON, if present
      compacted conversation messages
      latest user request

    The model does NOT choose user_id or the SQLite store.
    LangGraph injects them securely into tools.

17) TaskAgent reasoning model decides which tool is required
    Supervisor only selected the TaskAgent; supervisor did NOT create task steps.

    For the new request, TaskAgent LLM creates structured arguments such as:

      create_task(
          title="Project launch",
          steps=[
              "Prepare the launch checklist",
              "Deploy the project",
              "Notify stakeholders"
          ]
      )

    LLM CALL #2: TaskAgent selects create_task and creates title/steps

18) create_task tool executes --> tools/tasks.py:32-45
    LLM supplies: title, steps
    LangGraph injects: user_id, persistent store

    effective call is conceptually:

      create_task(
          title=...,
          steps=...,
          user_id=state["user_id"],
          store=smartdesk persistent store
      )

19) SmartDeskStore.create_task() --> memory/store.py:60-66
    generates a UUID task_id
    creates:

      {
        "task_id": "generated-id",
        "title": "Project launch",
        "status": "in_progress",
        "steps": [
          {"description": "...", "done": false},
          {"description": "...", "done": false},
          {"description": "...", "done": false}
        ]
      }

    persists it with:
      namespace = ("tasks", user_id)
      key       = task_id
      value     = complete task dictionary

    THIS is the permanent write to `smartdesk_memory.sqlite`.

20) create_task tool loads and returns the saved task --> tools/tasks.py:42-43
    internal TaskAgent receives a successful ToolMessage containing the real:
      task_id, title, status and steps

21) TaskAgent model runs again after seeing the ToolMessage
    it produces a final human-readable response, for example:

      Task created successfully.
      1. Prepare the launch checklist - Pending
      2. Deploy the project - Pending
      3. Notify stakeholders - Pending
      Reference: generated-task-id

    LLM CALL #3: TaskAgent converts persisted tool result into final response

22) agent.invoke() returns to run_agent() --> agents/common.py:149-153
    result messages contain:
      original input messages
      AIMessage(create_task tool call)
      ToolMessage(saved task dictionary)
      AIMessage(final TaskAgent response)

23) run_agent removes already-existing input messages --> agents/common.py:153

    messages = result["messages"][len(input_messages):]

    only newly generated TaskAgent messages are returned to the main graph
    so the user's HumanMessage is not duplicated

24) _unrecovered_tool_error() --> agents/common.py:66-80
    checks whether a tool failed and the agent failed to recover
    if no unrecovered failure exists, AgentRun(messages, error=None) is returned

25) _task_updates(run.messages) --> agents/task_agent.py:39-58, 89
    scans successful ToolMessages from:
      create_task
      get_task
      mark_step_done

    reads the structured task dictionary using task_id as the key
    for a new task it returns:
      tasks = [new persisted task]
      completed = []

26) task_agent_node builds its GlobalState patch --> agents/task_agent.py:90-98

    {
      "messages": newly generated agent/tool/final messages,
      "task_output": final human-readable TaskAgent response,
      "task_queue": [new active task]
    }

27) update_task_queue reducer --> state.py:21-29
    state.py:51 declares task_queue with this reducer

    reducer first indexes old active tasks by task_id:
      queue = {task_id: task}

    for each incoming task:
      status != "completed" -> insert/update queue[task_id]
      status == "completed" -> remove queue[task_id]

    New-task example:
      old task_queue = []
      incoming task  = project-launch task
      new task_queue = [project-launch task]

    Same task_id is updated rather than duplicated.

28) Task-only workflow finishes --> graph.py:300-304, 355-358
    _after_task() returns "finalize" because active_agent == "task"
    finalize_node() is a pass-through because workflow != "research_task"
    graph reaches END at graph.py:362

29) LangGraph checkpointer saves the updated GlobalState
    `smartdesk_checkpoints.sqlite` stores this data against thread_id:
      messages
      active_agent = "task"
      task_queue = current active task snapshot
      task_output = final TaskAgent text
      routing_log
      completed_tasks
      summary and other state fields

30) return ChatResponse(...) --> api.py:191-195
    answer = result["messages"][-1].content
    active_agent = "task"
    FastAPI serializes this object into JSON and sends it to the frontend

31) frontend receives response --> frontend/app.js:19
    bubble("assistant", data.answer, data.active_agent)
    the trailing `Reference: <task_id>` is shown as metadata by bubble()

32) (in background, AFTER the response)
    _maybe_summarize() --> api.py:24-32, 183-188
    if conversation exceeds 12 human turns, older messages are summarized
    task data in `smartdesk_memory.sqlite` is not deleted by summarization


# Memory: where task data is stored

1) `smartdesk_memory.sqlite` = permanent task source of truth
   scope: user_id
   namespace: ("tasks", user_id)
   key: task_id
   value: title, status and ordered steps

   A task created in thread A can still exist for the same user in thread B.

2) `smartdesk_checkpoints.sqlite` = thread conversation/graph state
   scope: thread_id
   stores messages, task_queue, completed_tasks, task_output and routing_log

3) `task_queue` = active-task snapshot in GlobalState
   used to understand short follow-ups like "next" and "done"
   it is not the authoritative permanent database record
   tools load the latest task from `smartdesk_memory.sqlite` before mutation

4) `task_output` = latest TaskAgent response text
   used for same-turn handoff to WriterAgent in multi-agent workflows
   cleared by begin_turn_node() at the next user turn

5) `completed_tasks` = completed task records accumulated in thread state
   when a task becomes completed, the reducer removes it from task_queue
   and task_agent_node adds it to completed_tasks

6) `messages` = user, AI tool-call, ToolMessage and final AI response history
   stored by the checkpointer and compacted/summarized when conversation grows


# "next" / "continue" flow for an active task

1) checkpoint loads a non-empty task_queue + HumanMessage("next")

2) supervisor_node() sees:
     task_queue exists
     normalized message is in _TASK_CONTINUATIONS
   it directly sets active_agent="task" with NO supervisor LLM call
   see graph.py:39-41 and 145-147

3) _task_context(state) adds active task JSON to context

4) run_agent gives TaskAgent:
     TASK_SYSTEM_PROMPT
     active task JSON
     compact conversation
     HumanMessage("next")

5) TaskAgent LLM calls get_task(exact persisted task_id)
   LLM CALL #1 for this turn

6) get_task --> tools/tasks.py:48-60
   LangGraph injects user_id and store
   SmartDeskStore.get_task() --> memory/store.py:68-73
   reads latest record from ("tasks", user_id), task_id

7) TaskAgent LLM reads the task and reports the next unfinished step
   it must NOT call mark_step_done because the user did not confirm completion
   LLM CALL #2 for this turn

8) database task and task_queue remain unchanged

9) final response returns through graph -> api.py -> frontend


# "done" / "mark done" flow for an active task

1) checkpoint loads a non-empty task_queue + HumanMessage("done")

2) supervisor fast-path directly selects TaskAgent
   NO supervisor routing LLM call

3) _task_context(state) supplies exact active task_id and structured steps

4) TaskAgent LLM calls get_task(task_id)
   LLM CALL #1 for this turn

5) get_task returns the latest permanent task as ToolMessage

6) TaskAgent LLM identifies exactly the next unfinished index and calls:
     mark_step_done(task_id, step_index)
   LLM CALL #2 for this turn

7) mark_step_done tool --> tools/tasks.py:63-79
   _task_was_loaded() first verifies a successful get_task ToolMessage exists
   direct mutation without loading the task is rejected

8) SmartDeskStore.mark_step_done() --> memory/store.py:75-95
   obtains a per-user/per-task lock
   reloads the latest permanent task
   validates step_index is in range
   validates it is exactly the next unfinished step
   sets that step's done=true
   sets status="completed" only when every step is done
   writes updated task back to `smartdesk_memory.sqlite`

9) TaskAgent LLM sees updated ToolMessage and writes final response
   LLM CALL #3 for this turn

10) _task_updates() extracts the updated task

11) if task remains in_progress:
      update_task_queue replaces the old snapshot with updated snapshot

12) if final step makes status="completed":
      update_task_queue removes the task from active task_queue
      task_agent_node adds the task to completed_tasks
      permanent task remains stored with status="completed"

13) updated GlobalState is checkpointed and final answer goes to frontend


# LLM call count

1) New task creation, normally 3 calls:
   supervisor router LLM
   TaskAgent LLM -> create_task
   TaskAgent LLM -> final response after tool result

2) "next" with an active task, normally 2 calls:
   supervisor uses fast-path, so 0 router calls
   TaskAgent LLM -> get_task
   TaskAgent LLM -> final next-step response

3) "done" with an active task, normally 3 calls:
   supervisor uses fast-path, so 0 router calls
   TaskAgent LLM -> get_task
   TaskAgent LLM -> mark_step_done
   TaskAgent LLM -> final response

4) New task + explicitly requested save_note, normally 4 calls:
   supervisor router LLM
   TaskAgent LLM -> create_task
   TaskAgent LLM -> save_note
   TaskAgent LLM -> final response

5) Additional calls are possible when:
   malformed provider tool call triggers the one allowed retry
   the agent needs an unexpected recovery step
   background summarization runs after more than 12 human turns

Python tools and SQLite operations are NOT LLM calls.


# Responsibility split

1) Supervisor LLM:
   decides which workflow/agent owns the request

2) TaskAgent LLM:
   creates task title and steps
   decides which task tool to call
   interprets structured tool results
   writes the final user-facing response

3) Deterministic Python tools:
   enforce required input schemas
   inject authenticated user_id and store
   create/load/update persistent task records

4) SmartDeskStore:
   enforces user-scoped keys
   enforces ordered step completion
   performs the actual SQLite writes

5) GlobalState reducers:
   append messages
   upsert/remove active task_queue records
   append completed task records

