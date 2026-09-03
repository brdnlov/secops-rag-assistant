"""Phase 6 — cumulative Anthropic credit ledger shared by the eval harnesses.

Prevents the session-8 incident where unguarded RAGAS judge runs exhausted the
$5 Anthropic credit. Every LLM-consuming run (generation + RAGAS judge) appends
its tracked spend to eval/results/credit_budget.json (gitignored), and a
--budget-usd ceiling halts a run *before it starts* if the projected cost
would exceed the remaining budget.

Spend is tracked, not guessed: the generator reports exact token counts from
its messages, and the RAGAS judge reports exact tokens via cost_cb driven off
the raw API usage. The ledger simply sums what each harness already knows.

The ledger file lives alongside the other eval artifacts (eval/results/) which
is already gitignored, so spend history never leaks into git.
"""
import json
import time
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[0] / "results"
LEDGER = RESULTS_DIR / "credit_budget.json"

# Default ceiling for a single session's *projected* spend. Deliberately below
# the full credit balance so an overrun can never zero out the account.
DEFAULT_BUDGET_USD = 2.0


def _empty_ledger():
    return {"total_spend_usd": 0.0, "runs": [], "updated_at": None}


def load_ledger():
    """Return the cumulative spend ledger (creating an empty one if absent)."""
    if not LEDGER.exists():
        return _empty_ledger()
    try:
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return _empty_ledger()


def total_spent():
    return float(load_ledger().get("total_spend_usd", 0.0))


def project_remaining(budget_usd):
    """Remaining budget = budget ceiling minus cumulative recorded spend."""
    return budget_usd - total_spent()


def record(name, spend_usd, details=None):
    """Append a run's spend to the ledger and persist.

    spend_usd: float, exact tracked spend for this run (sum of input + output
    at the model's token prices).
    details: optional dict of metadata (tokens, model, samples) for auditing.
    """
    ledger = load_ledger()
    entry = {
        "ts": time.time(),
        "run": name,
        "spend_usd": round(float(spend_usd), 6),
    }
    if details:
        entry["details"] = details
    ledger["runs"].append(entry)
    ledger["total_spend_usd"] = round(
        sum(float(r["spend_usd"]) for r in ledger["runs"]), 6
    )
    ledger["updated_at"] = entry["ts"]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    LEDGER.write_text(json.dumps(ledger, indent=2), encoding="utf-8")
    return ledger["total_spend_usd"]


def assert_budget(projected_usd, budget_usd, label):
    """Halt before a run starts if projected spend exceeds the remaining budget.

    projected_usd: float, the run's projected (not actual) cost.
    budget_usd: the session ceiling; None disables the guard entirely.
    """
    if budget_usd is None:
        return
    remaining = project_remaining(budget_usd)
    if projected_usd > remaining:
        raise SystemExit(
            f"[credit-budget] ABORT {label}: projected ${projected_usd:.3f} exceeds "
            f"remaining ${remaining:.3f} of the ${budget_usd:.2f} budget ceiling "
            f"(cumulative spend ${total_spent():.3f}). Raise --budget-usd or reduce "
            f"scope; nothing was spent."
        )


def print_status(budget_usd):
    ledge = load_ledger()
    print(f"[credit-budget] cumulative Anthropic spend: "
          f"${ledge['total_spend_usd']:.4f} across {len(ledge['runs'])} runs")
    if budget_usd is not None:
        print(f"[credit-budget] remaining of ${budget_usd:.2f} session budget: "
              f"${project_remaining(budget_usd):.4f}")
