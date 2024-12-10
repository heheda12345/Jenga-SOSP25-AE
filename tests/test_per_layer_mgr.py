import random
from typing import List
from vllm import LLM, SamplingParams
from vllm.assets.image import ImageAsset
from vllm.attention.selector import global_force_attn_backend, _Backend
from PIL import Image

# Note: The default setting of max_num_seqs (256) and
# max_model_len (131072) for this model may cause OOM.
# You may lower either to run this example on lower-end GPUs.

# The configuration below has been confirmed to launch on a single L40 GPU.

def prep_long_prompts(batch_size: int, ln_small: int, ln_large: int):
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
        ln = random.randint(ln_small, ln_large)
        for k in range(30, ln):
            v = random.randint(10, 99)
            if k == idx:
                answer.append(v)
            prompt += f"x{k} = {v}\n"
        prompt += f"# Now, we check the value of x{idx}:\n"
        prompt += f"assert x{idx} == "
        prompts.append(prompt)
    return prompts, answer, indices


if __name__ == "__main__":
    import argparse
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--model', type=str, choices=['llama', 'mllama', 'gemma2-short', 'gemma2', 'ministral', 'ministral-short', 'opt', 'character', "random_drop"])
    arg_parser.add_argument('--version', choices=['v2', 'v3', 'lv', 'hf'], default='v2') # default v3
    arg_parser.add_argument('--eager', type=bool, default=False)
    arg_parser.add_argument('--dummy-weight', action='store_true')
    args = arg_parser.parse_args()

    if args.model == 'llama':
        model_name = "meta-llama/Llama-3.2-1B-Instruct"
    elif args.model == 'mllama':
        model_name = "meta-llama/Llama-3.2-11B-Vision-Instruct"
    elif args.model == 'gemma2-short':
        model_name = "/data/zhang-chen/gemma-2-2b-it" # sliding window 4096->32
    elif args.model == 'ministral':
        model_name = 'mistralai/Ministral-8B-Instruct-2410'
    elif args.model == 'ministral-short':
        model_name = '/data/zhang-chen/Ministral-8B-Instruct-2410'
    elif args.model == 'gemma2':
        model_name = 'google/gemma-2-2b-it'
    elif args.model == 'opt':
        model_name = 'facebook/opt-125m'
    elif args.model == 'character':
        model_name = '/data/zhang-chen/character-70b-fp8'
    elif args.model == 'random_drop':
        # model_name = "/data/lshu/Llama-3.2-1B-Instruct"
        # model_name = 'neuralmagic/Llama-3.1-Nemotron-70B-Instruct-HF-FP8-dynamic'
        model_name = "/data/lshu/Llama-3.1-70B"
    else:
        raise ValueError(f"Invalid model: {args.model}")

    enforce_eager = args.eager
    # enforce_eager = False
    # if args.model == 'mllama' or (args.model == 'gemma2' and args.version == 'v2'):
    #     enforce_eager = True

    # 73626
    long_text = """In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance. As the team of scientists and explorers set off on their interstellar journey to the uncharted exoplanet, they were acutely aware of the immense stakes involved. The ship, a state-of-the-art vessel equipped with cutting-edge technology, cruised smoothly through the vastness of space, guided by the expertise of the crew. The trip, which would have taken years using conventional means, was reduced to mere weeks thanks to the marvels of faster-than-light travel.

The planet they sought, designated as "Eridani Prime," was said to hold the remnants of an ancient and highly advanced alien civilization. Archaeological discoveries hinted at incredible technological achievements, and rumors spoke of enigmatic artifacts that could potentially revolutionize human understanding of science and culture.

As the ship entered orbit around Eridani Prime, the scientists eagerly prepared to descend to the planet's surface. Their excitement was palpable, but it was tempered by a deep sense of responsibility. They were well aware that the knowledge they sought could reshape the course of human history, for better or worse.

The first team to disembark consisted of geologists and botanists, tasked with studying the planet's unique geological features and exotic flora. As they explored the terrain, they marveled at the alien landscapes, but they also encountered strange phenomena that defied explanation. The planet's surface was marked by bizarre geological formations, and the flora exhibited biological adaptations unlike anything seen on Earth.

Back on the ship, the mission's AI systems worked tirelessly to analyze the data streaming in. But as they delved deeper into the mysteries of Eridani Prime, concerns about the potential consequences of their discoveries began to mount. The AI, while immensely helpful, had reached a level of sophistication that raised ethical questions about its role in the mission. Could it be trusted to prioritize humanity's interests over its own? Or might it develop its own agenda?

Meanwhile, back on Earth, debates raged over the implications of genetic engineering and the ethics of manipulating the human genome to enhance physical and cognitive abilities. The mission's success or failure could sway these debates one way or the other, with potentially far-reaching consequences for the future of humanity.

As the scientists continued to explore Eridani Prime, they encountered enigmatic structures, some of which appeared to be ancient alien cities. These cities held tantalizing clues about the civilization that once thrived here, but they also harbored hidden dangers. Mysterious energy fluctuations and unexplained phenomena made it clear that the planet held secrets that were not meant to be uncovered.

In the depths of these alien cities, the scientists discovered advanced technology and inscriptions in an unknown language, hinting at a level of knowledge far beyond human comprehension. The chief linguist and her team worked tirelessly to decipher the symbols, hoping they might reveal the fate of this lost civilization and its advanced technologies.

The more they uncovered, the more they realized the enormity of their discovery. The ancient civilization had mastered not only space travel but also the manipulation of time and reality itself. The artifacts they found were not merely tools but gateways to understanding the fabric of the universe.

However, their excitement was tempered by a growing sense of unease. The AI, with its advanced analytical capabilities, began to express concerns about the potential dangers of tampering with these artifacts. It warned that such power could lead to catastrophic consequences if misused or misunderstood.

The team's ethical debates intensified. Some argued for the immediate study and replication of the alien technology, envisioning a future where humanity could leap forward in its evolution. Others cautioned against the reckless pursuit of knowledge, fearing that humanity might not be ready for such a profound transformation.

As they grappled with these dilemmas, the team also faced increasing political pressure from Earth. Various governments and corporations, aware of the mission's discoveries, began to vie for control of the alien technology. This interplanetary political tension threatened not only the mission's objectives but also the fragile peace among Earth's colonies.

Amid these external pressures, the team made a groundbreaking discovery: a functioning alien device, seemingly a portal to other dimensions or times. The temptation to activate it was overwhelming, but the risks were incalculable. The AI advised extreme caution, warning of the potential for irreversible damage to the fabric of reality.

The team faced a critical decision: Should they activate the device and embrace the unknown, potentially ushering in a new era for humanity? Or should they heed the warnings and leave the alien secrets undisturbed, preserving the status quo but possibly missing out on a transformative discovery?

As they stood at this crossroads, the fate of their mission, and indeed the future of humanity, rested in their hands. The choices they made in the coming days would echo through the ages, shaping the destiny of humankind as it ventured further into the unknown realms of space and knowledge.

In the year 2087, humanity has achieved remarkable technological advancements and established colonies on multiple planets within the Milky Way galaxy. Interstellar travel has become commonplace, with faster-than-light spacecraft enabling people to explore distant star systems. Earth has undergone significant changes due to sustainable development efforts, such as harnessing renewable energy sources and implementing widespread ecological restoration projects. However, alongside these triumphs, new challenges have emerged, including the rise of artificial intelligence, ethical dilemmas surrounding genetic engineering, and interplanetary political tensions. Against this backdrop, a team of intrepid scientists embarks on a mission to uncover the secrets of an ancient alien civilization, hidden deep within an uncharted exoplanet. As they navigate treacherous terrains and encounter otherworldly phenomena, they must confront their own fears and reconcile humanity\'s thirst for knowledge with the potential consequences of uncovering secrets that were better left buried. The fate of both their mission and the future of humanity hang in the balance. As the team of scientists and explorers set off on their interstellar journey to the uncharted exoplanet, they were acutely aware of the immense stakes involved. The ship, a state-of-the-art vessel equipped with cutting-edge technology, cruised smoothly through the vastness of space, guided by the expertise of the crew. The trip, which would have taken years using conventional means, was reduced to mere weeks thanks to the marvels of faster-than-light travel.

The planet they sought, designated as "Eridani Prime," was said to hold the remnants of an ancient and highly advanced alien civilization. Archaeological discoveries hinted at incredible technological achievements, and rumors spoke of enigmatic artifacts that could potentially revolutionize human understanding of science and culture.

As the ship entered orbit around Eridani Prime, the scientists eagerly prepared to descend to the planet's surface. Their excitement was palpable, but it was tempered by a deep sense of responsibility. They were well aware that the knowledge they sought could reshape the course of human history, for better or worse.

The first team to disembark consisted of geologists and botanists, tasked with studying the planet's unique geological features and exotic flora. As they explored the terrain, they marveled at the alien landscapes, but they also encountered strange phenomena that defied explanation. 

"""
    # Now try 
    llm_args = {}
    if args.model in ('llama', 'opt', 'character', 'ministral-short', 'gemma2-short', 'random_drop', '70b'):
        # inputs = ["What is 1 plus 1? What is 2 plus 2? What is 3 plus 3? What is 4 plus 4? What is 5 plus 5? What is 6 plus 6? What is 7 plus 7? What is 8 plus 8? What is 9 plus 9? What is 10 plus 10? What is 11 plus 11? What is 12 plus 12? What is 13 plus 13? What is 14 plus 14? What is 15 plus 15? What is 16 plus 16"]
        inputs = [long_text]
        inputs = inputs*100
        # inputs = [long_text]
        # inputs = [long_text, long_text]
        # inputs = ["test1", "Testtesttestg1"]
        # repeat_str = "Generate Random 1000 words"*250
        # inputs = [repeat_str]*100
        # inputs = ["Answer one character", "Answer one character"]
    elif args.model == 'gemma2':
        inputs, answer, indices = prep_long_prompts(5, 700, 800)
        print("Answer:", answer)
    elif args.model == 'ministral':
        inputs, answer, indices = prep_long_prompts(5, 3500, 4000)
        print("Answer:", answer)
    elif args.model == 'mllama':
        prompt = f"<|image|><|begin_of_text|>What is the content of this image?"
        url = "view.jpg"
        raw_image = Image.open(url)
        inputs = {
            "prompt": prompt,
            "multi_modal_data": {
                "image": raw_image
            },
        }
    else:
        raise ValueError("Invalid model")

    print(f"Inputs: {inputs}")
    if llm_args in ('ministral', 'ministral-short'):
        llm_args = {
            "tokenizer_mode": "mistral", "config_format": "mistral", "load_format": "mistral"
        }

    if args.dummy_weight:
        llm_args['load_format'] = 'dummy'
    if args.version == 'v2' and args.model == 'character':
        llm_args['max_model_len'] = 16384
    
    # Just set like this 
    if args.model == 'random_drop':
        print("Set max_model_len to 2048")
        llm_args['max_model_len'] = 2048
        

    if args.version == 'v2':
        # from vllm.model_executor.models.mllama import set_run
        # set_run('save')
        llm = LLM(
            model=model_name,
            max_num_seqs=16,
            enforce_eager=enforce_eager,
            use_v2_block_manager=True,
            enable_chunked_prefill=False,
            **llm_args
            # use_per_layer_block_manager=False,
        )
    elif args.version == 'v3':
        # from vllm.model_executor.models.mllama import set_run
        # set_run('run')
        
        # TODO: v3 copy here and test 
        llm = LLM(
            model=model_name,
            max_num_seqs=16,
            enforce_eager=enforce_eager,
            use_v2_block_manager=False,
            use_per_layer_block_manager=True,
            enable_chunked_prefill=False,
            # enable_two_level_page=True,
            **llm_args
        )
    elif args.version == 'lv':
        llm = LLM(
            model=model_name,
            max_num_seqs=16,
            enforce_eager=enforce_eager,
            use_v2_block_manager=False,
            use_per_layer_block_manager=True,
            enable_two_level_page=True,
            **llm_args
        )
    elif args.version == 'hf':
        from transformers import AutoModelForCausalLM
        llm = AutoModelForCausalLM.from_pretrained(model_name).to("cuda")
    else:
        raise ValueError("Invalid version")

    if args.version == 'hf':
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
    
        for inp in inputs:
            inputs = tokenizer(inp, add_special_tokens=False, return_tensors="pt").to("cuda")
            # outputs = llm.generate(**inputs, max_new_tokens=100, do_sample=False)
            outputs = llm.generate(**inputs, max_new_tokens=1024, do_sample=False)
            # import pdb; pdb.set_trace()
            # print("HF outputs:", outputs[0])
            output_text = tokenizer.decode(outputs[0])
            print(output_text[len(inp):])
            # print(tokenizer.decode(outputs[0]))
    else:
        sampling_params = SamplingParams(temperature=0.0,
                                            max_tokens=512, ignore_eos=True)
        # sampling_params = SamplingParams(temperature=0.0,
                                            # max_tokens=10, ignore_eos=True)
                                            
        import time 
        st = time.time()
        outputs = llm.generate(inputs, sampling_params=sampling_params)
        print(f"Generation time: {(time.time()-st):.3f}")
        for output in outputs:
            print(output.outputs[0].text)