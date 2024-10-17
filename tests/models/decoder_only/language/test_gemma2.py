import random
from typing import List

import pytest

from vllm import LLM, SamplingParams

models = [
    'google/gemma-2-9b-it',
]


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("max_tokens", [32])
@pytest.mark.parametrize("vllm_engine_args", [{
    'max_model_len': 8192,
    'enforce_eager': True,
}])
def test_gemma2_model(
    hf_runner,
    vllm_runner,
    model: str,
    max_tokens: int,
    vllm_engine_args,
) -> None:
    prompts, answer, indices = prep_prompts(5)

    with vllm_runner(model, dtype="bfloat16",
                     **vllm_engine_args) as vllm_model:
        check_window(prompts)(vllm_model.model)
        vllm_outputs = vllm_model.generate_greedy(prompts, max_tokens)
    vllm_output_strs = []
    for prompt, o in zip(prompts, vllm_outputs):
        assert o[1].startswith(prompt)
        vllm_output_strs.append(o[1][len(prompt):])
    print("vllm_outputs:", vllm_output_strs)

    with hf_runner(model, dtype="bfloat16") as hf_model:
        hf_outputs = hf_model.generate_greedy(prompts, max_tokens)
    hf_output_strs = []
    for prompt, o in zip(prompts, hf_outputs):
        assert o[1].startswith(prompt)
        hf_output_strs.append(o[1][len(prompt):])
    print("hf_outputs:", hf_output_strs)
    print("check hf answer")
    check_answers(indices, answer, hf_output_strs)
    print("check vllm answer")
    check_answers(indices, answer, vllm_output_strs)


def prep_prompts(batch_size: int):
    """
    Generate prompts which a bunch of assignments,
    then asking for the value of one of them.
    The prompt is just under 10k tokens; sliding window is 4k
    so the answer is outside sliding window, but should still be correct.
    """
    prompts: List[str] = []
    answer: List[int] = []
    indices: List[int] = []
    random.seed(1)
    for _ in range(batch_size):
        idx = random.randint(30, 90)
        indices.append(idx)
        prompt = "```python\n# We set a number of variables, " + \
                 f"x{idx} will be important later\n"
        ln = random.randint(700, 800)
        for k in range(30, ln):
            v = random.randint(10, 99)
            if k == idx:
                answer.append(v)
            prompt += f"x{k} = {v}\n"
        prompt += f"# Now, we check the value of x{idx}:\n"
        prompt += f"assert x{idx} == "
        prompts.append(prompt)
    return prompts, answer, indices


def check_answers(indices: List[int], answer: List[int], outputs: List[str]):
    answer2 = [int(text[0:2].strip()) for text in outputs]
    print(list(zip(indices, zip(answer, answer2))))
    numok = 0
    for a1, a2 in zip(answer, answer2):
        if a1 == a2:
            numok += 1
    frac_ok = numok / len(answer)
    print(f"Num OK: {numok}/{len(answer)} {frac_ok}")
    assert frac_ok == 1


def check_window(prompts: List[str]):

    def inner(llm: LLM):
        sliding_window = llm.llm_engine.model_config.get_sliding_window()
        assert sliding_window and sliding_window > 0
        print(
            "length",
            [len(llm.get_tokenizer().tokenize(prompt)) for prompt in prompts])
        assert any(
            len(llm.get_tokenizer().tokenize(prompt)) > sliding_window
            for prompt in prompts)

    return inner
