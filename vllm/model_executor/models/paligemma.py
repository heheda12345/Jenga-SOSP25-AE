from typing import (Iterable, List, Literal, Mapping, Optional, Tuple,
                    TypedDict, Union)

import torch
from torch import nn
from transformers import PaliGemmaConfig

from vllm.attention import AttentionMetadata
from vllm.attention.backends.utils import MMEmbeddingMetadata
from vllm.config import CacheConfig, ModelConfig, MultiModalConfig, ParallelConfig, SchedulerConfig
from vllm.core.block_v3.custom_block import SelfAttentionManager, SlidingWindowManager, VisionEmbeddingManager
from vllm.core.block_v3.registry import BLOCK_MANAGER_REGISTRY
from vllm.inputs import INPUT_REGISTRY, InputContext, LLMInputs
from vllm.logger import init_logger
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.sampler import SamplerOutput
# from vllm.model_executor.models.gemma import GemmaForCausalLM
from vllm.model_executor.models.gemma2 import Gemma2ForCausalLM
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.base import MultiModalInputs
from vllm.multimodal.utils import cached_get_tokenizer
from vllm.sequence import IntermediateTensors

from .interfaces import SupportsMultiModal, SupportsPP
from .siglip import (SiglipVisionModel, dummy_image_for_siglip,
                     dummy_seq_data_for_siglip, get_max_siglip_image_tokens)
from .utils import AutoWeightsLoader, merge_multimodal_embeddings

logger = init_logger(__name__)


def custom_block_manager_for_paligemma(model_config: ModelConfig,
                                       cache_config: CacheConfig,
                                       parallel_config: ParallelConfig,
                                       scheduler_config: SchedulerConfig):
    custom_managers = {}
    sliding_window = model_config.hf_config.get_text_config().sliding_window
    print("model config sliding window", sliding_window)
    share_layer_ids = set()
    for i in range(model_config.get_num_layers(parallel_config)):
        if i % 2 == 0 and sliding_window is not None:
            custom_managers[str(i)] = SlidingWindowManager(
                model_config, parallel_config, cache_config.cache_dtype,
                cache_config.block_size, sliding_window)
        else:
            custom_managers[str(i)] = SelfAttentionManager(
                model_config, parallel_config, cache_config.cache_dtype,
                cache_config.block_size)
            share_layer_ids.add(str(i))
    custom_managers["vision"] = VisionEmbeddingManager(
        model_config, parallel_config, cache_config.cache_dtype,
        cache_config.block_size, share_layer_ids)
    return custom_managers


class PaliGemmaImagePixelInputs(TypedDict):
    type: Literal["pixel_values"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, num_channels, height, width)`"""


class PaliGemmaImageEmbeddingInputs(TypedDict):
    type: Literal["image_embeds"]
    data: torch.Tensor
    """Shape: `(batch_size * num_images, image_feature_size, hidden_size)`

    `hidden_size` must match the hidden size of language model backbone.
    """


PaliGemmaImageInputs = Union[PaliGemmaImagePixelInputs,
                             PaliGemmaImageEmbeddingInputs]


def get_max_paligemma_image_tokens(ctx: InputContext):
    hf_config = ctx.get_hf_config(PaliGemmaConfig)
    vision_config = hf_config.vision_config

    return get_max_siglip_image_tokens(vision_config)


def dummy_data_for_paligemma(ctx: InputContext, seq_len: int,
                             mm_counts: Mapping[str, int]):
    hf_config = ctx.get_hf_config(PaliGemmaConfig)
    vision_config = hf_config.vision_config
    num_images = mm_counts["image"]

    seq_data = dummy_seq_data_for_siglip(
        vision_config,
        seq_len,
        num_images,
        image_token_id=hf_config.image_token_index,
    )

    mm_data = dummy_image_for_siglip(vision_config, num_images)
    mm_data['text'] = seq_data.prompt_token_ids
    return seq_data, mm_data


def input_processor_for_paligemma(ctx: InputContext, llm_inputs: LLMInputs):

    """
    The correct prompt format needs to be:
    '<image>' * image_feature_size + '<bos>' + prompt + '\n'

    See https://github.com/huggingface/transformers/blob/25245ec26dc29bcf6102e1b4ddd0dfd02e720cf5/src/transformers/models/paligemma/processing_paligemma.py#L55
    """ # noqa

    multi_modal_data = llm_inputs.get("multi_modal_data")
    if multi_modal_data is None or "image" not in multi_modal_data:
        return llm_inputs

    model_config = ctx.model_config
    hf_config = ctx.get_hf_config(PaliGemmaConfig)

    tokenizer = cached_get_tokenizer(model_config.tokenizer)
    image_feature_size = hf_config.text_config.num_image_tokens
    image_token_str = tokenizer.decode(hf_config.image_token_index)
    bos_token = tokenizer.decode(hf_config.bos_token_id)
    image_token_str_pad = image_token_str * image_feature_size
    image_token_ids_pad = [hf_config.image_token_index] * image_feature_size

    orig_prompt = llm_inputs.get("prompt")
    orig_prompt_ids = llm_inputs.get("prompt_token_ids")

    if orig_prompt is not None and image_token_str in orig_prompt:
        logger.warning(
            "The image token '%s' was detected in the prompt and "
            "will be removed. Please follow the proper prompt format"
            " documented on HuggingFace.", image_token_str)
        orig_prompt = orig_prompt.replace(image_token_str, "")
        orig_prompt_ids.remove(hf_config.image_token_index)

    new_prompt = f"{image_token_str_pad}{bos_token}{orig_prompt}\n"
    new_token_ids = image_token_ids_pad + orig_prompt_ids + [108]  #newline
    multi_modal_data['text'] = new_token_ids

    # NOTE: Create a defensive copy of the original inputs
    return LLMInputs(prompt_token_ids=new_token_ids,
                     prompt=new_prompt,
                     multi_modal_data=multi_modal_data)


class PaliGemmaMultiModalProjector(nn.Module):

    def __init__(self, vision_hidden_size: int, projection_dim: int):
        super().__init__()

        self.linear = nn.Linear(vision_hidden_size, projection_dim, bias=True)

    def forward(self, image_features: torch.Tensor) -> torch.Tensor:
        hidden_states = self.linear(image_features)
        return hidden_states


@MULTIMODAL_REGISTRY.register_image_input_mapper()
@MULTIMODAL_REGISTRY.register_max_image_tokens(get_max_paligemma_image_tokens)
@INPUT_REGISTRY.register_dummy_data(dummy_data_for_paligemma)
@INPUT_REGISTRY.register_input_processor(input_processor_for_paligemma)
@BLOCK_MANAGER_REGISTRY.register_block_manager(
    custom_block_manager_for_paligemma)
class PaliGemmaForConditionalGeneration(nn.Module, SupportsMultiModal,
                                        SupportsPP):

    def __init__(self,
                 config: PaliGemmaConfig,
                 multimodal_config: MultiModalConfig,
                 cache_config: Optional[CacheConfig] = None,
                 quant_config: Optional[QuantizationConfig] = None) -> None:
        super().__init__()

        self.config = config
        self.multimodal_config = multimodal_config

        self.vision_tower = SiglipVisionModel(config.vision_config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(
            vision_hidden_size=config.vision_config.hidden_size,
            projection_dim=config.vision_config.projection_dim)

        self.quant_config = quant_config
        self.language_model = Gemma2ForCausalLM(config.text_config,
                                                cache_config, quant_config)
        logit_scale = getattr(config, "logit_scale", 1.0)
        self.language_model.logits_processor.scale *= logit_scale

        self.make_empty_intermediate_tensors = (
            self.language_model.make_empty_intermediate_tensors)

    @property
    def sampler(self):
        return self.language_model.sampler

    def _validate_pixel_values(self, data: torch.Tensor) -> torch.Tensor:
        h = w = self.config.vision_config.image_size
        expected_dims = (3, h, w)
        actual_dims = tuple(data.shape[1:])

        if actual_dims != expected_dims:
            expected_expr = ("batch_size", *map(str, expected_dims))
            raise ValueError(
                f"The expected shape of pixel values is {expected_expr}. "
                f"You supplied {tuple(data.shape)}.")

        return data

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[PaliGemmaImageInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        image_embeds = kwargs.pop("image_embeds", None)

        if pixel_values is None and image_embeds is None:
            return None

        if pixel_values is not None:
            if not isinstance(pixel_values, torch.Tensor):
                raise ValueError("Incorrect type of pixel values. "
                                 f"Got type: {type(pixel_values)}")

            # Remove the N dimension until multiple images are supported.
            pixel_values = pixel_values.squeeze(1)

            return PaliGemmaImagePixelInputs(
                type="pixel_values",
                data=self._validate_pixel_values(pixel_values),
            )

        if image_embeds is not None:
            if not isinstance(image_embeds, torch.Tensor):
                raise ValueError("Incorrect type of image embeddings. "
                                 f"Got type: {type(image_embeds)}")

            # Remove the N dimension until multiple images are supported.
            image_embeds = image_embeds.squeeze(1)

            return PaliGemmaImageEmbeddingInputs(
                type="image_embeds",
                data=image_embeds,
            )

        raise AssertionError("This line should be unreachable.")

    def _image_pixels_to_features(
        self,
        vision_tower: SiglipVisionModel,
        pixel_values: torch.Tensor,
    ) -> torch.Tensor:

        target_dtype = vision_tower.get_input_embeddings().weight.dtype
        image_features = vision_tower(pixel_values.to(dtype=target_dtype))

        return image_features

    def _process_image_input(
        self,
        image_input: PaliGemmaImageInputs,
    ) -> torch.Tensor:

        if image_input["type"] == "image_embeds":
            return image_input["data"]

        assert self.vision_tower is not None
        pixel_values = image_input["data"]
        image_features = self._image_pixels_to_features(
            self.vision_tower,
            pixel_values,
        )

        return self.multi_modal_projector(image_features)

    def forward(self,
                input_ids: torch.Tensor,
                positions: torch.Tensor,
                kv_caches: List[torch.Tensor],
                attn_metadata: AttentionMetadata,
                intermediate_tensors: Optional[IntermediateTensors] = None,
                **kwargs: object) -> Union[SamplerOutput, IntermediateTensors]:
        if intermediate_tensors is not None:
            input_ids = None
            inputs_embeds = None
        else:
            vision_metadata = attn_metadata['vision']
            assert isinstance(vision_metadata, MMEmbeddingMetadata)
            if vision_metadata.num_mm_prefills > 0:
                # run vision encoder
                vision_kwargs = MultiModalInputs.as_kwargs(
                    vision_metadata.multi_modal_inputs,
                    device=input_ids.device)
                image_input = self._parse_and_validate_image_input(
                    **vision_kwargs)
                mm_input_ids = vision_metadata.mm_token_ids
                assert image_input is not None
                inputs_embeds = self.language_model.model.get_input_embeddings(
                    mm_input_ids)
                vision_embeddings = self._process_image_input(image_input)
                # https://github.com/huggingface/transformers/blob/main/src/transformers/models/paligemma/modeling_paligemma.py#L294 # noqa
                vision_embeddings = vision_embeddings * (
                    self.config.hidden_size**-0.5)
                inputs_embeds = merge_multimodal_embeddings(
                    mm_input_ids, inputs_embeds, vision_embeddings,
                    self.config.image_token_index)
                if kv_caches['vision'].numel() > 0:
                    kv_caches['vision'][
                        vision_metadata.mm_slot_mapping, :inputs_embeds.
                        shape[-1]] = inputs_embeds
            if vision_metadata.num_prefills > 0:
                # fetch prefill embeddings from kv_caches and run decode embeddings
                num_prefill_tokens = vision_metadata.num_prefill_tokens
                prefill_slots = vision_metadata.slot_mapping[:
                                                             num_prefill_tokens]
                hidden_size = self.config.text_config.hidden_size
                if kv_caches['vision'].numel() == 0:
                    # dummy embedding for profile run
                    prefill_embeds = torch.ones(
                        (num_prefill_tokens, hidden_size),
                        dtype=kv_caches['vision'].dtype,
                        device=input_ids.device)
                else:
                    # fetch prefill embeddings from kv_caches
                    prefill_embeds = kv_caches['vision'][
                        prefill_slots, :hidden_size]
                decode_embeds = self.language_model.model.get_input_embeddings(
                    input_ids[num_prefill_tokens:])
                inputs_embeds = torch.cat((prefill_embeds, decode_embeds),
                                          dim=0)
                input_ids = None
            else:
                inputs_embeds = None

        hidden_states = self.language_model.model(input_ids,
                                                  positions,
                                                  kv_caches,
                                                  attn_metadata,
                                                  intermediate_tensors,
                                                  inputs_embeds=inputs_embeds)

        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[torch.Tensor]:
        return self.language_model.compute_logits(hidden_states,
                                                  sampling_metadata)

    def sample(
        self,
        logits: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        return self.language_model.sample(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        loader = AutoWeightsLoader(self)
        loader.load_weights(weights)
