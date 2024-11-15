import pytest
from vllm import LLM
from vllm.sampling_params import SamplingParams
from vllm.utils import Device


def test_gemma_eviction():
    model_name = '/data/zhang-chen/gemma-2-2b-it'

    inputs = [
        "What is 1 plus 1? What is 2 plus 2? What is 3 plus 3? What is 4 plus 4? What is 5 plus 5? What is 6 plus 6? What is 7 plus 7? What is 8 plus 8? What is 9 plus 9? What is 10 plus 10? What is 11 plus 11? What is 12 plus 12? What is 13 plus 13? What is 14 plus 14? What is 15 plus 15? What is 16 plus 16?"
    ]

    inputs2 = [
        "What is 1 plus 1? What is 2 plus 2? What is 3 plus 3? What is 4 plus 4? What is 5 plus 5? What is 6 plus 6? What is 7 plus 7? What is 8 plus 8? What is 9 plus 9? What is 10 plus 10? What is 11 plus 11? What is 12 plus 12? What is 13 plus 13? What is 14 plus 14? What is 16 plus 16? What is 17 plus 17?"
    ]

    llm = LLM(
        model=model_name,
        max_num_seqs=16,
        use_v2_block_manager=False,
        enable_prefix_caching=True,
        disable_log_stats=False,
        use_per_layer_block_manager=True,
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=20,
    )
    outputs = llm.generate(inputs, sampling_params=sampling_params)
    outputs = llm.generate(inputs2, sampling_params=sampling_params)

    evictor = llm.llm_engine.scheduler[
        0].block_manager.custom_block_manager.global_block_allocator._allocators[
            Device.GPU].evictor
    r'''
    # the allocated blocks are: (N refers to NULL block)
    sliding window:  0- 1- 2- 3- 4- 5- 6- 7- 8-19-21 (input1)
                    N- N- N/           \22-21-25-27 (input2)
    self attention:  9-10-11-12-13-14-15-16-17-20-22 (input1)
                                        \23-24-26-28 (input2)
    '''
    evict_order = [
        2, 1, 0, 4, 3, 5, 6, (19, 20), (8, 17), (7, 16), (25, 26), (21, 24),
        (22, 23), 15, 14, 13, 12, 11, 10, 9
    ]  # (a, b) means a and b can be evicted in arbitrary order
    evict_time = {}
    for t, blocks in enumerate(evict_order):
        if isinstance(blocks, int):
            evict_time[blocks] = t
        else:
            for block in blocks:
                evict_time[block] = t

    evict_result = []
    for i in range(26):
        evict_result.append(evictor.evict()[0])
    print("evict_result:", evict_result)

    for old, new in zip(evict_result[:-1], evict_result[1:]):
        assert evict_time[old] <= evict_time[new]

    with pytest.raises(ValueError):
        evictor.evict()
