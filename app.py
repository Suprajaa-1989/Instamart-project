
import os
import sys

if not os.path.exists("data/instamart.db"):
    sys.exit("Data not found. Run `python run_all.py` first.")
if not os.path.exists("mock_llm.py"):                              # same stand-in fallback as run_all.py
    sys.path.insert(0, os.path.abspath("standins"))

from bot.server import main  # noqa: E402

main()
