import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=360, stage="base-b", runner_config="2-gpu-large")


class TestKimiLinear(CustomTestCase):
    extra_args = []

    @classmethod
    def setUpClass(cls):
        cls.model = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--tp-size", "2", "--trust-remote", *cls.extra_args],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["score"], 0.88)


class TestKimiLinearUnifiedMemory(TestKimiLinear):
    """Same GSM8K bar under --enable-unified-memory: the KDA state and the MLA
    latent KV share one unified byte buffer (the attention backend defaults to
    triton under the flag; the mamba-radix hook pins page_size=1 and
    non-overlap scheduling)."""

    extra_args = ["--enable-unified-memory"]


class TestKimiLinearNgramChain(TestKimiLinear):
    """NGRAM speculative decoding, CHAIN drafts (bfs breadth 1): the minimal
    KDA target-verify configuration — multi-token verify over the KDA
    recurrent state with per-step intermediate capture, no tree walk. NGRAM is
    target-only (no draft model / draft KV), so this exercises exactly the KDA
    verify branch + the family-generic commit scatter. Same GSM8K bar as
    spec-off: verify must be output-lossless."""

    extra_args = [
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-ngram-max-bfs-breadth",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


class TestKimiLinearNgramTree(TestKimiLinear):
    """NGRAM speculative decoding with the default TREE drafts (bfs breadth
    10, 12 draft tokens): adds the conv ancestor walk + per-branch SSM parent
    reload on top of the chain path."""

    extra_args = ["--speculative-algorithm", "NGRAM"]


class TestKimiLinearNgramChainUnifiedMemory(TestKimiLinear):
    """NGRAM chain under --enable-unified-memory: the KDA intermediate verify
    states live in the relocatable spec-state band and the verify indices go
    through the virtual->physical translate; the MLA verify read rail runs
    the unified ragged translate at page_size=1."""

    extra_args = [
        "--enable-unified-memory",
        "--speculative-algorithm",
        "NGRAM",
        "--speculative-ngram-max-bfs-breadth",
        "1",
        "--speculative-num-draft-tokens",
        "4",
    ]


class TestKimiLinearNgramTreeUnifiedMemory(TestKimiLinear):
    """NGRAM tree under --enable-unified-memory (tree at page_size=1 is the
    supported tree configuration under the unified pool)."""

    extra_args = ["--enable-unified-memory", "--speculative-algorithm", "NGRAM"]


if __name__ == "__main__":
    unittest.main()
