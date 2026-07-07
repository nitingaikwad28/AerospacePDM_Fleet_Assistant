# chat_cli.py
# Terminal chat interface for the Fleet Assistant. Run from inside fleet_assistant/:
#   python chat_cli.py

import llm_client
import agent


def main():
    llm_up = llm_client.is_available()
    print("=" * 60)
    print("FLEETGUARD AI - Conversational Fleet Assistant")
    print("=" * 60)
    if llm_up:
        print(f"LLM backend: connected (Ollama, model configured in config.py)")
    else:
        print("LLM backend: not reachable - using deterministic template answers.")
        print("(Install/run Ollama to get natural-language phrasing; the assistant")
        print(" works fully without it, just with plainer wording.)")
    print()
    print("Ask about the fleet, e.g.:")
    print('  "give me a fleet summary"')
    print('  "what needs attention this week"')
    print('  "why is AC-014 flagged"')
    print('  "has this happened before on other landing_gear"')
    print('  "work order for AC-014"')
    print("Type 'quit' to exit.\n")

    while True:
        try:
            message = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not message:
            continue
        if message.lower() in ("quit", "exit"):
            print("Goodbye.")
            break

        result = agent.answer(message)
        print(f"\nAssistant [{result['source']}]: {result['text']}\n")


if __name__ == "__main__":
    main()
