from transformers import AutoTokenizer

from rusty_dawg import (
    DiskDawg,
    PyDawg,
)

# =========================================================
# fixed path
# =========================================================

DISK_DAWG_PATH = (
    "./rusty/diskdawg_photochat"
)

LOCAL_TOKENIZER_DIR = (
    "./rusty/models--gpt2/"
    "snapshots/"
    "11c5a3d58116272d56d2b232146260c555324687"
)

# =========================================================
# novelty description calculator
# =========================================================

class NGramNoveltyCalculator:

    def __init__(
        self,
        dawg_path,
        tokenizer_dir,
    ):

        self.tokenizer = (
            AutoTokenizer.from_pretrained(
                tokenizer_dir,
                local_files_only=True
            )
        )

        if self.tokenizer.pad_token is None:

            self.tokenizer.pad_token = (
                self.tokenizer.eos_token
            )

        self.dawg = DiskDawg.load(
            dawg_path
        )

        self.py_dawg = PyDawg(
            self.dawg,
            self.tokenizer
        )

    def calculate_text_novelty(
        self,
        text,
        n=2,
    ):

        text = text.strip()

        if not text:

            return 0.0

        suffix_context = (
            self.py_dawg.get_suffix_context(
                text
            )
        )

        tokens = suffix_context[
            "tokens"
        ]

        nnsl_vector = suffix_context[
            "suffix_contexts"
        ]

        total_tokens = len(tokens)

        if total_tokens < n:

            return 0.0

        total_ngrams = (
            total_tokens - n + 1
        )

        count_nnsl_less_than_n = sum(
            1
            for x in nnsl_vector
            if x < n
        )

        novel_ngrams = max(
            0,
            count_nnsl_less_than_n - (n - 1)
        )

        score = (
            novel_ngrams / total_ngrams
            if total_ngrams > 0
            else 0.0
        )

        return float(score)

# =========================================================
# load novelty description model
# =========================================================

def load_novelty_description_model():

    calculator = (
        NGramNoveltyCalculator(
            dawg_path=DISK_DAWG_PATH,
            tokenizer_dir=LOCAL_TOKENIZER_DIR,
        )
    )

    return calculator

# =========================================================
# novelty description score
# =========================================================

def novelty_description_score(
    model,
    description,
    n_gram=2,
):

    description = (
        description
        .replace("|", ".")
        .strip()
    )

    score = (
        model.calculate_text_novelty(
            text=description,
            n=n_gram,
        )
    )

    return float(score)