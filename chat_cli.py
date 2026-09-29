"""Optional command-line version for quick testing."""
from rag.pipeline import BOT_NAME, DuoBot

bot = DuoBot()
history = []
print(f"{BOT_NAME} ready. Type 'exit' to quit.")
while True:
    q = input("You: ").strip()
    if q.lower() in ("exit", "quit"):
        break
    ans, docs, standalone, route = bot.ask(q, history)
    print(f"{BOT_NAME}: {ans}\n")
    history += [("user", q), ("assistant", ans)]
