"""
Builds an instruction-tuning dataset for code review.

Each example teaches the model:
    INPUT:  a code diff (or snippet)
    OUTPUT: a structured review — issues found, severity, suggested fix

This dataset contains 35 hand-written, diverse examples covering common
real-world bug categories (SQL injection, null checks, resource leaks,
race conditions, mutable defaults, N+1 queries, etc.) PLUS clean-code
examples with no issues, so the model learns to recognize good code too
instead of always flagging something.

Want even better results? Extend this list with real examples from your
own past PRs or open-source repos you know well — the more real, varied
examples, the better the fine-tuned model generalizes.
"""

import json
import random
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "review_dataset.jsonl"

SYSTEM_PROMPT = (
    "You are an expert code reviewer. Given a code diff, identify real "
    "issues (bugs, security problems, style violations, missing tests). "
    "For each issue, output: severity (low/medium/high), a one-line "
    "description, and the line(s) affected. If there are no issues, say so."
)

PROMPT_TEMPLATE = """### Instruction:
{system_prompt}

### Diff:
{diff}

### Review:
"""


def format_example(diff: str, review: str) -> dict:
    return {
        "text": PROMPT_TEMPLATE.format(system_prompt=SYSTEM_PROMPT, diff=diff)
        + review
    }


EXAMPLES = [
    # --- SQL injection ---
    {
        "diff": '+ def get_user(user_id):\n+     query = f"SELECT * FROM users WHERE id = {user_id}"\n+     return db.execute(query)\n',
        "review": "- severity: high\n  description: SQL injection — user input interpolated directly into query string.\n  line: 2\n  suggested_fix: use parameterized queries, e.g. db.execute('SELECT * FROM users WHERE id = ?', (user_id,))\n",
    },
    {
        "diff": '+ def search_products(name):\n+     sql = "SELECT * FROM products WHERE name LIKE \'%" + name + "%\'"\n+     cursor.execute(sql)\n',
        "review": "- severity: high\n  description: SQL injection via string concatenation in LIKE clause.\n  line: 2\n  suggested_fix: use parameterized queries with bound parameters instead of concatenation.\n",
    },
    # --- Division / arithmetic ---
    {
        "diff": "+ def average(nums):\n+     return sum(nums) / len(nums)\n",
        "review": "- severity: medium\n  description: division by zero if nums is empty; no input validation.\n  line: 2\n  suggested_fix: add a guard clause, e.g. 'if not nums: raise ValueError(...)'\n",
    },
    {
        "diff": "+ def percentage(part, total):\n+     return (part / total) * 100\n",
        "review": "- severity: medium\n  description: division by zero if total is 0.\n  line: 2\n  suggested_fix: check total != 0 before dividing, or return 0/None explicitly.\n",
    },
    # --- Loop / commit batching ---
    {
        "diff": "+ for item in items:\n+     results.append(process(item))\n+     db.commit()\n",
        "review": "- severity: medium\n  description: committing inside a loop is inefficient and can leave the DB partially committed on failure.\n  line: 3\n  suggested_fix: batch commits after the loop, wrapped in try/except with rollback.\n",
    },
    # --- Mutable default argument ---
    {
        "diff": "+ def add_item(item, items=[]):\n+     items.append(item)\n+     return items\n",
        "review": "- severity: medium\n  description: mutable default argument — the same list is reused across calls, causing unexpected shared state.\n  line: 1\n  suggested_fix: use 'items=None' and initialize inside the function with 'if items is None: items = []'\n",
    },
    # --- Broad exception handling ---
    {
        "diff": "+ try:\n+     process_payment(order)\n+ except:\n+     pass\n",
        "review": "- severity: high\n  description: bare except clause silently swallows all exceptions, including ones that should stop payment processing.\n  line: 3\n  suggested_fix: catch specific exceptions and log or handle them appropriately, never silently pass on payment logic.\n",
    },
    {
        "diff": "+ try:\n+     data = json.loads(response.text)\n+ except Exception as e:\n+     data = {}\n",
        "review": "- severity: medium\n  description: catching all exceptions and silently defaulting hides real parsing/network errors.\n  line: 2\n  suggested_fix: catch json.JSONDecodeError specifically and log the error before defaulting.\n",
    },
    # --- Hardcoded secrets ---
    {
        "diff": '+ API_KEY = "sk-live-4f8a9b2c1d3e"\n+ client = Client(api_key=API_KEY)\n',
        "review": "- severity: high\n  description: hardcoded API key committed directly in source code.\n  line: 1\n  suggested_fix: load from environment variable or a secrets manager, e.g. os.environ['API_KEY']\n",
    },
    # --- Null / None checks ---
    {
        "diff": "+ def get_email(user):\n+     return user.profile.email\n",
        "review": "- severity: medium\n  description: no null check on user.profile before accessing .email; will raise AttributeError if profile is None.\n  line: 2\n  suggested_fix: add 'if user.profile is None: return None' or use getattr with a default.\n",
    },
    {
        "diff": "+ def get_first_item(items):\n+     return items[0]\n",
        "review": "- severity: medium\n  description: no check for an empty list; will raise IndexError.\n  line: 2\n  suggested_fix: check 'if not items: return None' before indexing.\n",
    },
    # --- Off-by-one ---
    {
        "diff": "+ for i in range(1, len(items)):\n+     total += items[i]\n",
        "review": "- severity: medium\n  description: loop starts at index 1, skipping items[0] in the sum.\n  line: 1\n  suggested_fix: start the range at 0 unless skipping the first item is intentional.\n",
    },
    # --- Resource leaks ---
    {
        "diff": "+ def read_config(path):\n+     f = open(path)\n+     return f.read()\n",
        "review": "- severity: medium\n  description: file handle is never closed, leaking a file descriptor.\n  line: 2\n  suggested_fix: use a context manager, e.g. 'with open(path) as f: return f.read()'\n",
    },
    {
        "diff": "+ conn = connect_to_db()\n+ cursor = conn.cursor()\n+ cursor.execute(query)\n+ return cursor.fetchall()\n",
        "review": "- severity: low\n  description: database connection is never closed after use.\n  line: 1\n  suggested_fix: use a context manager or explicitly close conn/cursor in a finally block.\n",
    },
    # --- Type / comparison bugs ---
    {
        "diff": "+ def is_admin(role):\n+     if role == 'Admin':\n+         return True\n",
        "review": "- severity: low\n  description: case-sensitive string comparison may miss valid roles like 'admin' or 'ADMIN'.\n  line: 2\n  suggested_fix: normalize case, e.g. 'if role.lower() == \"admin\":'\n",
    },
    {
        "diff": "+ def is_valid(count):\n+     if count != None:\n+         return True\n",
        "review": "- severity: low\n  description: comparing to None with '!=' instead of 'is not None' — works but is non-idiomatic and can behave unexpectedly with custom __eq__ methods.\n  line: 2\n  suggested_fix: use 'if count is not None:'\n",
    },
    # --- Input validation ---
    {
        "diff": "+ def create_user(name, age):\n+     user = User(name=name, age=age)\n+     user.save()\n+     return user\n",
        "review": "- severity: medium\n  description: no validation on 'age' (could be negative or non-numeric) before saving.\n  line: 2\n  suggested_fix: validate inputs before constructing the object, e.g. raise ValueError if age < 0.\n",
    },
    # --- Race conditions / concurrency ---
    {
        "diff": "+ if not os.path.exists(lockfile):\n+     with open(lockfile, 'w') as f:\n+         f.write('locked')\n",
        "review": "- severity: high\n  description: check-then-act race condition — another process could create the lockfile between the exists() check and the open() call (TOCTOU bug).\n  line: 1\n  suggested_fix: use 'open(lockfile, 'x')' which atomically fails if the file already exists, instead of check-then-create.\n",
    },
    # --- Logging sensitive data ---
    {
        "diff": '+ logger.info(f"User login attempt: {username}, password: {password}")\n',
        "review": "- severity: high\n  description: logging plaintext password — sensitive data exposure in logs.\n  line: 1\n  suggested_fix: never log passwords or other secrets; log only the username and outcome.\n",
    },
    # --- Missing tests mention ---
    {
        "diff": "+ def calculate_discount(price, percent):\n+     return price - (price * percent / 100)\n",
        "review": "- severity: low\n  description: no test coverage visible for edge cases like percent > 100 or negative price.\n  line: 2\n  suggested_fix: add unit tests covering percent=0, percent=100, and negative inputs.\n",
    },
    # --- N+1 query pattern ---
    {
        "diff": "+ for order in orders:\n+     customer = db.query(Customer).filter_by(id=order.customer_id).first()\n+     print(customer.name)\n",
        "review": "- severity: medium\n  description: N+1 query pattern — fetches a Customer individually inside a loop instead of a single batched query.\n  line: 2\n  suggested_fix: fetch all needed customers in one query up front, e.g. using a JOIN or an IN clause, then look them up in memory.\n",
    },
    # --- Unused imports / dead code ---
    {
        "diff": "+ import requests\n+ import json\n+\n+ def load_data(path):\n+     with open(path) as f:\n+         return json.load(f)\n",
        "review": "- severity: low\n  description: 'requests' is imported but never used in this function.\n  line: 1\n  suggested_fix: remove the unused import.\n",
    },
    # --- Clean code examples (no issues) ---
    {
        "diff": "+ def divide(a, b):\n+     if b == 0:\n+         raise ValueError('Cannot divide by zero')\n+     return a / b\n",
        "review": "- severity: none\n  description: no issues found — division by zero is properly guarded against.\n",
    },
    {
        "diff": "+ def get_user_email(user):\n+     if user is None or user.profile is None:\n+         return None\n+     return user.profile.email\n",
        "review": "- severity: none\n  description: no issues found — null checks are properly handled before attribute access.\n",
    },
    {
        "diff": "+ def read_config(path):\n+     with open(path) as f:\n+         return json.load(f)\n",
        "review": "- severity: none\n  description: no issues found — file is properly closed via context manager.\n",
    },
    {
        "diff": "+ def get_user(user_id):\n+     query = 'SELECT * FROM users WHERE id = ?'\n+     return db.execute(query, (user_id,))\n",
        "review": "- severity: none\n  description: no issues found — query is properly parameterized, preventing SQL injection.\n",
    },
    {
        "diff": "+ try:\n+     value = int(user_input)\n+ except ValueError:\n+     logger.warning(f'Invalid input: {user_input}')\n+     value = 0\n",
        "review": "- severity: none\n  description: no issues found — specific exception is caught and logged appropriately.\n",
    },
    {
        "diff": "+ def add_item(item, items=None):\n+     if items is None:\n+         items = []\n+     items.append(item)\n+     return items\n",
        "review": "- severity: none\n  description: no issues found — mutable default argument pitfall is correctly avoided.\n",
    },
    {
        "diff": "+ API_KEY = os.environ.get('API_KEY')\n+ if not API_KEY:\n+     raise RuntimeError('API_KEY environment variable not set')\n",
        "review": "- severity: none\n  description: no issues found — secret is loaded from environment, not hardcoded, with a clear failure message if missing.\n",
    },
    {
        "diff": "+ def get_first_item(items):\n+     if not items:\n+         return None\n+     return items[0]\n",
        "review": "- severity: none\n  description: no issues found — empty list case is handled before indexing.\n",
    },
    # --- More variety: multiple issues in one diff ---
    {
        "diff": "+ def process_order(order_id):\n+     query = f\"SELECT * FROM orders WHERE id = {order_id}\"\n+     order = db.execute(query)\n+     return order.total / order.item_count\n",
        "review": (
            "- severity: high\n  description: SQL injection via f-string interpolation.\n  line: 2\n  suggested_fix: use parameterized queries.\n"
            "- severity: medium\n  description: potential division by zero if item_count is 0.\n  line: 4\n  suggested_fix: guard against item_count == 0 before dividing.\n"
        ),
    },
    {
        "diff": "+ def update_inventory(product_id, quantity):\n+     product = Product.objects.get(id=product_id)\n+     product.stock -= quantity\n+     product.save()\n",
        "review": "- severity: medium\n  description: no check that stock does not go negative after subtracting quantity.\n  line: 3\n  suggested_fix: validate 'quantity <= product.stock' before subtracting, or clamp at zero.\n",
    },
    {
        "diff": "+ def send_notification(user, message):\n+     if user.email:\n+         send_email(user.email, message)\n+     if user.phone:\n+         send_sms(user.phone, message)\n",
        "review": "- severity: none\n  description: no issues found — both notification channels are properly guarded with existence checks.\n",
    },
    {
        "diff": "+ class Cache:\n+     data = {}\n+     def set(self, key, value):\n+         self.data[key] = value\n",
        "review": "- severity: medium\n  description: 'data' is a class-level mutable attribute, shared across all instances of Cache instead of being per-instance.\n  line: 2\n  suggested_fix: initialize 'self.data = {}' inside '__init__' instead.\n",
    },
    {
        "diff": "+ def get_discounted_price(price, discount_pct):\n+     if discount_pct < 0 or discount_pct > 100:\n+         raise ValueError('discount_pct must be between 0 and 100')\n+     return price * (1 - discount_pct / 100)\n",
        "review": "- severity: none\n  description: no issues found — input is validated before use.\n",
    },
]


def load_seed_examples() -> list[dict]:
    """Returns the curated example set defined above."""
    return EXAMPLES


def main():
    seeds = load_seed_examples()
    examples = [format_example(s["diff"], s["review"]) for s in seeds]

    random.shuffle(examples)

    with OUTPUT_PATH.open("w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(examples)} examples to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
