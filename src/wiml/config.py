"""Parameters transcribed from the original notebook (cell 3)."""

from dataclasses import dataclass


@dataclass
class Config:
    model_id: str = "Qwen/Qwen3-4B"
    output_dir: str = "reproduction/full"
    smoke: bool = False
    candidate_pool_smoke: int = 60
    candidate_pool_full: int = 2500
    target_examples_smoke: int = 20
    target_examples_full: int = 150
    use_4bit: bool = True
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_new_tokens_direct: int = 64
    max_new_tokens_think: int = 256
    decode_seeds_smoke: tuple = (17,)
    decode_seeds_full: tuple = (17, 29, 41)
    eval_seed: int = 20260811
    bootstrap_samples: int = 5000
    # The original independent evaluation uses 150 facts even in smoke mode.
    independent_examples: int = 150
    # Revisions were unpinned in the source. Optional pins support future runs.
    model_revision: str | None = None
    dataset_revision: str | None = None

    @property
    def candidate_pool(self):
        return self.candidate_pool_smoke if self.smoke else self.candidate_pool_full

    @property
    def target_examples(self):
        return self.target_examples_smoke if self.smoke else self.target_examples_full

    @property
    def decode_seeds(self):
        return self.decode_seeds_smoke if self.smoke else self.decode_seeds_full
