"""Interactive SmartDesk command-line interface."""
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from memory.checkpointer import runtime, thread_config, user_id
from observability import log

logger = log("cli")


def main() -> None:
    """Run an isolated user and thread conversation."""
    from graph import build_graph
    load_dotenv(".env")

    username = input("Username: ").strip()
    if not username:
        return

    owner = user_id(username)
    config = thread_config(owner)
    print("Thread: main (automatic; resumes for this username)")

    with runtime() as (store, saver):
        app = build_graph(store, saver)

        while text := input("You: ").strip():
            try:
                print("SmartDesk: Working...", flush=True)
                logger.info("request started")

                result = app.invoke(
                    {"messages": [HumanMessage(text)], "user_id": owner},
                    config,
                )

                logger.info("request finished")
                print("SmartDesk:", result["messages"][-1].content)

            except KeyboardInterrupt:
                print("\nCancelled.")
                break

            except Exception as exc:
                logger.exception("request failed")
                print(f"SmartDesk: Sorry, the request failed ({type(exc).__name__}). Please try again.")
