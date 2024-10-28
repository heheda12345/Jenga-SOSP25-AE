from vllm.config import ModelConfig, ParallelConfig
from vllm.core.block_v3.custom_block import SelfAttentionManager, SlidingWindowManager

model_id = 'google/gemma-2-2b-it'


def test_hit_len_self_attn():
    model_config = ModelConfig(
        model_id,
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="bfloat16",
        revision=None,
    )
    parallel_config = ParallelConfig(1, 1, False)
    self_attn_mgr = SelfAttentionManager(model_config, parallel_config,
                                         'bfloat16', 16)

    query1 = [False] * 16  # noqa
    out1 = self_attn_mgr.get_possible_hit_lens(query1)
    assert out1 == [(0, 0)]

    query2 = [True] * 16 + [False]  # noqa
    out2 = self_attn_mgr.get_possible_hit_lens(query2)
    assert out2 == [(0, 16 * 16)]

    query3 = [True] * 16 + [False] * 16  # noqa
    out3 = self_attn_mgr.get_possible_hit_lens(query3)
    assert out3 == [(0, 16 * 16)]


def test_hit_len_sliding_window():
    model_config = ModelConfig(
        model_id,
        model_id,
        tokenizer_mode="auto",
        trust_remote_code=False,
        seed=0,
        dtype="bfloat16",
        revision=None,
    )
    parallel_config = ParallelConfig(1, 1, False)
    sliding_window_mgr = SlidingWindowManager(model_config, parallel_config,
                                              'bfloat16', 16, 32)
    # minimum hit is 3 blocks

    query1 = [False] * 16  # noqa
    out1 = sliding_window_mgr.get_possible_hit_lens(query1)
    assert out1 == [(0, 0)]

    query2 = [True] + [False] * 10  # noqa
    out2 = sliding_window_mgr.get_possible_hit_lens(query2)
    assert out2 == [(0, 16)]

    query3 = [True] * 3 + [False] * 16  # noqa
    out3 = sliding_window_mgr.get_possible_hit_lens(query3)
    assert out3 == [(0, 3 * 16)]

    query4 = [False] + [True] * 2 + [False] * 16  # noqa
    out4 = sliding_window_mgr.get_possible_hit_lens(query4)
    assert out4 == [(0, 0)]

    query5 = [False] + [True] * 3 + [False] * 16  # noqa
    out5 = sliding_window_mgr.get_possible_hit_lens(query5)
    assert out5 == [(0, 0), (4 * 16, 4 * 16)]

    query6 = [False] + [True] * 4 + [False] * 16  # noqa
    out6 = sliding_window_mgr.get_possible_hit_lens(query6)
    assert out6 == [(0, 0), (4 * 16, 5 * 16)]
