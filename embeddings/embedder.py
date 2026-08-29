"""Phase 3 — Voyage AI voyage-4 embedder.

Embeds corpus chunks into 1024-dim dense vectors with a cost guard and
token accounting (the Voyage client reports per-batch total_tokens).

Hand-rolled batching (no langchain) so the mechanics stay visible, matching
the project convention. The price constants come from AGENTS.md:

  - model:      voyage-4
  - dimensions: 1024
  - price:      $0.06 / 1M tokens

Cost guard: halt if cumulative spend exceeds MAX_TOTAL_SPEND ($5). At
voyage-4 that is ~83M tokens — far beyond our corpus (~1-2M tokens), but it
catches a runaway re-embed loop during eval iterations.

The embedder is intentionally storage-agnostic: it returns embeddings plus
token accounting, and leaves checkpointing/upsert to the Qdrant indexer
(embeddings/indexer.py).
"""

import time

from pathlib import Path

from dotenv import load_dotenv

# Load VOYAGE_API_KEY from the repo-local .env (gitignored, never committed).
ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

PRICE_PER_MTOK = 0.06  # USD per 1M tokens (voyage-4)
MAX_TOTAL_SPEND = 5.0  # USD hard ceiling
MAX_TOKENS_AT_GUARD = int(MAX_TOTAL_SPEND / (PRICE_PER_MTOK / 1_000_000))
BATCH_SIZE = 8  # worst-case 8 x ~1K-token GDPR articles stays under the 10K TPM window

# Voyage free-tier limits (no payment method on the account): 3 requests/minute
# and 10K tokens/minute. Pacing does the math below; the throttling lesson from
# the first run was that the SDK's own retries (one 429 for every re-sent
# attempt) double/triple request rate and keep the RPM bucket empty forever, so
# the client is deliberately created with max_retries=0 and ALL rate-limit
# handling lives here: one HTTP call per batch, spaced to stay inside the
# limits, with a full-window wait on a 429.
RATE_LIMIT_RPM = 3
RATE_LIMIT_TPM = 10_000
RATE_LIMIT_RETRY_WAIT = 65  # seconds to wait on a 429 when the API sends no retry-after
RATE_LIMIT_RETRIES = 8  # attempts per batch before giving up (run is resumable)
SPACING_HEADROOM = 1.2  # extra margin so the 10K TPM rolling window isn't blown
MIN_INTER_BATCH_SLEEP = 45.0  # one request per >=45s keeps the 3 RPM bucket full

MODEL = "voyage-4"
EMBEDDING_DIM = 1024


class EmbedCostGuardError(RuntimeError):
    """Raised when cumulative embedding spend exceeds the cost ceiling."""


class Embedder:
    def __init__(self, client=None, model=MODEL, batch_size=BATCH_SIZE,
                 max_total_spend=MAX_TOTAL_SPEND):
        import voyageai  # imported lazily so non-embedding code can import this module

        self._voyage = voyageai
        # max_retries=0 is deliberate: the SDK's tenacity retries re-send the
        # request on a 429, which consumes the already-scarce RPM budget without
        # waiting long enough to matter. Rate-limit handling is in _embed_batch.
        self.client = client or voyageai.Client(max_retries=0)
        self.model = model
        self.batch_size = batch_size
        self.max_total_spend = max_total_spend
        self.total_tokens = 0
        self.total_spend = 0.0

    def _tokens_to_spend(self, tokens):
        return tokens * (PRICE_PER_MTOK / 1_000_000)

    def _charge(self, tokens):
        """Add token cost to the running total; raise the guard if over ceiling."""
        self.total_tokens += tokens
        self.total_spend += self._tokens_to_spend(tokens)
        if self.total_spend > self.max_total_spend:
            raise EmbedCostGuardError(
                f"Cost guard exceeded: ${self.total_spend:.4f} > "
                f"${self.max_total_spend:.2f} ({self.total_tokens} tokens)"
            )

    def throttle_sleep(self, tokens):
        """Seconds to pause after a batch so the rolling 60s window stays under
        both free-tier limits: <3 calls and <=10K tokens per minute."""
        tpm_delay = tokens / (RATE_LIMIT_TPM / 60.0) * SPACING_HEADROOM
        return max(MIN_INTER_BATCH_SLEEP, tpm_delay)

    def _embed_batch(self, texts, input_type):
        """One API call with free-tier-friendly retry on rate limiting.

        The SDK does no retrying (max_retries=0). On a 429 we wait out a full
        rolling window (honoring retry-after when the API sends it) and retry,
        up to RATE_LIMIT_RETRIES attempts. Prior batches already hit Qdrant
        (indexer upserts per batch), so throwing after a sustained hold is
        always resumable from the next batch.
        """
        for attempt in range(1, RATE_LIMIT_RETRIES + 1):
            try:
                return self.client.embed(texts=texts, model=self.model,
                                         input_type=input_type)
            except self._voyage.error.RateLimitError as e:
                if attempt == RATE_LIMIT_RETRIES:
                    raise
                headers = getattr(e, "headers", None) or {}
                retry_after = headers.get("retry-after")
                wait = max(RATE_LIMIT_RETRY_WAIT, (int(retry_after) + 2 if retry_after else 0))
                print(f"[embedder] rate-limited (attempt {attempt + 1}/"
                      f"{RATE_LIMIT_RETRIES}); waiting {wait}s")
                time.sleep(wait)

    def embed(self, texts, input_type="document"):
        """Embed a list of texts as voyage-4 vectors.

        Returns a list of 1024-dim lists, one per input text, charging each
        batch's token count against the cost guard and pacing calls inside the
        free-tier rate windows.
        """
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            result = self._embed_batch(batch, input_type)
            tokens = int(getattr(result, "total_tokens", 0) or 0)
            self._charge(tokens)
            vectors.extend(list(v) for v in result.embeddings)
            if i + self.batch_size < len(texts):
                time.sleep(self.throttle_sleep(tokens))
        return vectors

    def cost_summary(self):
        return {
            "model": self.model,
            "total_tokens": self.total_tokens,
            "total_spend_usd": round(self.total_spend, 6),
            "guard_usd": self.max_total_spend,
        }
