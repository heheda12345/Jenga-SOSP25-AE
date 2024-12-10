"""Attention backend utils"""
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Dict, List, Type, TypeVar, Union

import numpy as np
import torch
import random 
import math 
import copy
from vllm.attention import (AttentionMetadata, AttentionMetadataBuilder,
                            AttentionState)
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.core.topk_utils import cache_topk, cache_topk_wo_none

import random 

random.seed(4)

if TYPE_CHECKING:
    from vllm.worker.model_runner_base import ModelRunnerBase

# Error string(s) for encoder/decoder
# unsupported attention scenarios
STR_NOT_IMPL_ENC_DEC_ROCM_HIP = ("ROCm/HIP is not currently supported "
                                 "with encoder/decoder models.")

PAD_SLOT_ID = -1

# Switch to numpy implementation of compute_slot_mapping
# if we have at least this many elements. Could be tuned further.
_COMPUTE_SLOT_MAPPING_NUMPY_NUMEL = 256

if TYPE_CHECKING:
    from vllm.worker.model_runner import ModelInputForGPUBuilder


def is_block_tables_empty(block_tables: Union[None, Dict]):
    """
    Check if block_tables is None or a dictionary with all None values.
    """
    if block_tables is None:
        return True
    return (isinstance(block_tables, dict)
            and all(value is None for value in block_tables.values()))


def compute_slot_mapping_start_idx(is_prompt: bool, query_len: int,
                                   context_len: int, sliding_window: int,
                                   use_v2_block_manager: bool):
    """
    Compute the start index of slot mapping.
    """
    start_idx = 0
    if is_prompt and sliding_window is not None:
        assert use_v2_block_manager or context_len == 0, (
            "Prefix caching is currently not supported with "
            "sliding window attention in V1 block manager")
        # When prefill, we use it to not write slots to kv cache
        # to save memory.
        start_idx = max(0, query_len - sliding_window)
    return start_idx


# def _compute_slot_mapping_python(slot_mapping: List[int],
#                                  block_table: List[int], range_start: int,
#                                  range_end: int, block_size: int):
#     for i in range(range_start, range_end):
#         block_number = block_table[i // block_size]
#         block_offset = i % block_size
#         slot = block_number * block_size + block_offset
#         slot_mapping.append(slot)

def compute_slot_mapping(is_profile_run: bool, slot_mapping: List[int],
                         seq_id: int, seq_len: int, context_len: int,
                         start_idx: int, block_size: int,
                         block_tables: Dict[int, List[int]]):
    """
    Compute slot mapping.
    """
    if is_profile_run:
        # During memory profiling, the block tables are not
        # initialized yet. In this case, we just use a dummy
        # slot mapping.
        # In embeddings, the block tables are {seq_id: None}.
        slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        return

    # Mask the [0, start_idx) tokens of the prompt with
    # PAD_SLOT_ID, where start_idx is max(0, seq_len -
    # sliding_window). For example, if the prompt len is 10,
    # sliding window is 8, and block size is 4, the first two
    # tokens are masked and the slot mapping will be
    # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    padding_mask_len = max(0, start_idx - context_len)
    slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)

    range_start = max(start_idx, context_len)
    range_end = seq_len
    numel = range_end - range_start
    block_table = block_tables[seq_id]

    # numpy implementation will be faster than python if we have
    # many elements, otherwise it will be slower.
    if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL:
        _compute_slot_mapping_python(slot_mapping, block_table, range_start,
                                     range_end, block_size)
    else:
        _compute_slot_mapping_numpy(slot_mapping, block_table, range_start,
                                    range_end, block_size)


def _compute_slot_mapping_python(block_table: List[int], range_start: int,
                                 range_end: int, block_size: int,
                                 is_prompt: bool,
                                 seq_id: int,
                                 num_layers: int) -> Dict[int, List[int]]:
    """
    This function should compute the slot mapping for seq {seq_id}.
    Return a dictionary of per-layer slot mappings. 
    """
    # print(f"Seq id: {seq_id}, is prompt: {is_prompt}, block table: {block_table}")
    # print(f"Compute slot mapping python, range start: {range_start}, range end: {range_end}")   
    # print(f"Block table (length: {len(block_table)}): {block_table}, block table size: {block_size}")
    # print(f"Is prompt: {is_prompt}, seq id: {seq_id}")
    
    if not is_prompt and seq_id in cache_topk_wo_none:
        max_topk_this_round = max(cache_topk_wo_none[seq_id])
    else:
        max_topk_this_round = None
    
    calc_slot_mapping = []
    for i in range(range_start, range_end):
        block_number = block_table[(i // block_size) % len(block_table)] 
        block_offset = i % block_size
        
        if is_prompt: 
            slot = block_number * block_size + block_offset
            # print(f"Seq id: {seq_id}, Slot: {slot}, block number: {block_number}, block offset: {block_offset}")
        else:
            assert range_end - range_start == 1, f"Range end - range start: {range_end - range_start} should be 1"
            block_number = block_table[-1] # NOTE: is this true? 
            assert max_topk_this_round is not None, f"Max top k this round should not be None, but got {max_topk_this_round}"
            residual = max_topk_this_round % block_size
            # block_offset = residual - 1 
            # random sample one slot as offset 
            block_offset_random = random.randint(0, max(residual-1, 0)) 
            slot = block_number * block_size + block_offset_random
            # print(f"Seq id: {seq_id}, Slot: {slot}, block number: {block_number}, block offset random: {block_offset_random}, residual: {residual}")
            # just take residual?
        # slot_mapping.append(slot)
        calc_slot_mapping.append(slot)
    
    slot_mapping_dict = {i: calc_slot_mapping for i in range(num_layers)}
    return slot_mapping_dict    
    
def _compute_slot_mapping_numpy(slot_mapping: List[int],
                                block_table: List[int], range_start: int,
                                range_end: int, block_size: int):
    block_table_array = np.array(block_table)
    idx = np.arange(range_start, range_end)
    block_offset = idx % block_size
    idx //= block_size
    seq_slot_mapping_array = block_table_array[idx]
    seq_slot_mapping_array *= block_size
    seq_slot_mapping_array += block_offset
    slot_mapping.extend(seq_slot_mapping_array)

def clean_func(
        recent_attn_weights_idx_last_shape: Any,
        past_kv_seq_lens_idx: Any,
        next_decoder_cache_seen_tokens: Any,
        schedule_gen_decay_ratio: Any,
        exceed_length_to_compress: Any,
        recent_length: Any,
        gen_compress_ratio: Any,
    ) -> None:
        last_shape = recent_attn_weights_idx_last_shape + 1    
        past_kv_seq_len = past_kv_seq_lens_idx
        current_kv_seq_len = next_decoder_cache_seen_tokens
                
        if current_kv_seq_len - recent_length - past_kv_seq_len >= exceed_length_to_compress: 
            if 1 + recent_length + exceed_length_to_compress > last_shape:
                keep_last_shape = last_shape - (1 + recent_length)
            else:
                keep_last_shape = exceed_length_to_compress
            
            context_length = keep_last_shape * gen_compress_ratio
            topk = max(int(context_length * schedule_gen_decay_ratio), 1)        
            return topk
        else:
            return None # Keep All
    
def update_parameter(recent_attn_weights_last_shape_list, layer_id, recent_length, exceed_length_to_compress, topk):
    # print(f"Recent attn weights last shape: {recent_attn_weights_last_shape_list[layer_id]}, recent length: {recent_length}, exceed length to compress: {exceed_length_to_compress}")
    
    x = recent_attn_weights_last_shape_list[layer_id] + 1
    
    if x - (1 + recent_length + exceed_length_to_compress) < 0:
        first_factor = 0 
    else:
        first_factor = x - (1 + recent_length + exceed_length_to_compress)
        
    recent_length_updated_to = first_factor + topk + 1 + recent_length
    return recent_length_updated_to



def compute_slot_mapping_top_k(is_profile_run: bool, slot_mappings: List[List[int]],
                         num_layers: int,
                         seq_id: int, seq_len: int, context_len: int,
                         start_idx: int, block_size: int,
                         block_tables: Dict[int, List[int]]):
    """
    Given this sequence ID, and all the K (num of tokens) it needs to retain 
    in each layer, allocate the slot mapping accordingly. 
    
    # Now self.slot_mapping is a List[List[int]] where 
    # - outer list is per layer
    # - inner list is per token [seq_1_t1_slot, seq_1_t2_slot, ..., seq_2_t1_slot, seq_2_t2_slot, ...]
            
    Return a slot mapping per layer. 
    """
    
    # if seq_id in cache_topk and len(cache_topk[seq_id]) > 0:
    #     print(f"Cache top k in utils looks like: {cache_topk}")
    #     assert len(cache_topk[seq_id]) == num_layers, f"Cache top k length: {len(cache_topk)}, cache top k: {cache_topk} does not match num_layers: {num_layers}"
    # else:
    #     print(f"Cache topk looks like this: {cache_topk} right now, seq id {seq_id} is not there")
        
    # print(f"Compute slot mapping for number of layers: {num_layers}, seq_len: {seq_len}, context len: {context_len}")
    if is_profile_run:
        # During memory profiling, the block tables are not
        # initialized yet. In this case, we just use a dummy
        # slot mapping.
        # In embeddings, the block tables are {seq_id: None}.
        # slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        # NOTE: not sure if this is correct 
        # Populate PAD_SLOT_ID for the num_layers        
        if len(slot_mappings) == 0:
            element = [PAD_SLOT_ID] * seq_len
            slot_mappings = [element for _ in range(num_layers)]
        else: 
            slot_mappings[0] = slot_mappings[0] + [PAD_SLOT_ID] * seq_len
            slot_mappings = [slot_mappings[0] for _ in range(num_layers)]
        
        # print(f"Shape of slot_mapping after profile run: {len(slot_mappings)}")
        return slot_mappings

    # Mask the [0, start_idx) tokens of the prompt with
    # PAD_SLOT_ID, where start_idx is max(0, seq_len -
    # sliding_window). For example, if the prompt len is 10,
    # sliding window is 8, and block size is 4, the first two
    # tokens are masked and the slot mapping will be
    # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    # HACK: This is a temporary fix for the issue where the 
    # NOTE: I don't think this part of the code is used 
    padding_mask_len = max(0, start_idx - context_len)
    slot_mappings.extend([PAD_SLOT_ID] * padding_mask_len)

    # TODO: Not sure if this part needs to be changed 
    range_start = max(start_idx, context_len)
    range_end = seq_len
    numel = range_end - range_start
    block_table = block_tables[seq_id]
    is_decode_stage = int(context_len) != 0
    
    # NOTE: random drop logic for decode stage 
    # Select a random position to place new token (drop the old token)
    
    # numpy implementation will be faster than python if we have
    # many elements, otherwise it will be slower.
    slot_mapping_dict_for_seq = _compute_slot_mapping_python(block_table, range_start,
                                 range_end, block_size, not is_decode_stage, seq_id, num_layers)
    
    # NOTE: is this going to be very slow with pythom implementation?
    # if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL:
    #     slot_mapping_dict_for_seq = _compute_slot_mapping_python(block_table, range_start,
    #                              range_end, block_size, not is_decode_stage, seq_id, num_layers)
    # else:
    #     slot_mapping_dict_for_seq = _compute_slot_mapping_numpy_top_k(block_table, range_start, range_end, \
    #                              block_size, not is_decode_stage, seq_id, num_layers)
        
        # raise NotImplementedError("Numpy implementation not supported yet")
    # else:
    #     _compute_slot_mapping_numpy(slot_mapping, block_table, range_start,
    #                                 range_end, block_size)
    
    # slot_mappings = [copy.deepcopy(slot_mapping) for _ in range(num_layers)]

    if len(slot_mappings) != 0:
        assert len(slot_mappings) == len(slot_mapping_dict_for_seq), f"Slot mapping length: {len(slot_mappings)} does not match slot mapping dict length: {len(slot_mapping_dict_for_seq)}"
    else:
        # Initialize slot mappings 
        for layer_id in range(num_layers):
            slot_mappings.append([])
        
    if len(cache_topk[seq_id]) != 0:    
        # do nothing 
        pass
        # print("*"*25 + "Start Compute Slot Mapping" + "*"*25)
        # print(f"Slot mapping length: {len(slot_mappings)}, Context length: {context_len}, Sequence length: {seq_len}")
        # print(f"Start index: {start_idx}, range start: {range_start}, range_end: {range_end}")
        # if is_decode_stage:
        #     print(f"Seq id: {seq_id}. DECODE STAGE NOW")
        #     # print(f"Slot mapping looks like this: {slot_mappings}")
        # else:
        #     print(f"Seq id: {seq_id}. PREFILL STAGE NOW")
        #     # print(f"Slot mapping looks like: {slot_mappings}")
        # print("-->" + "Random Drop")  
    else:
        # Directly extend each element of slot mapping with the slot mapping dict
        for layer_id in range(len(slot_mappings)):
            slot_mappings[layer_id].extend(slot_mapping_dict_for_seq[layer_id])
        
        return slot_mappings

    for layer_id in range(num_layers):
        # Get the top K value for this layer
        topk = cache_topk[seq_id][layer_id]
        
        ### Prefill Stage ### 
        if not is_decode_stage:
            # SELECT TOP K Tokens to keep
            if topk is not None: 
                # just keep topK of the slot mappings
                slot_mapping_dict_for_seq[layer_id] = slot_mapping_dict_for_seq[layer_id][:topk]
                # Mark the rest as -1 if any?
                slot_mapping_dict_for_seq[layer_id] = slot_mapping_dict_for_seq[layer_id] + [PAD_SLOT_ID] * (seq_len - topk)
                
        ### Decode Stage ###
        # NOTE: do we need to do anything here?
        
    # Now extend the slot mappings with the slot mapping dict
    for layer_id in range(len(slot_mappings)):
        slot_mappings[layer_id].extend(slot_mapping_dict_for_seq[layer_id])
    
    # if is_decode_stage:
    #     print(f"Slot mapping after decode stage modification: {slot_mappings}")
    # else:
    #     print(f"Slot mapping after prefill stage modification: {slot_mappings}")
    return slot_mappings
                         
def compute_slot_mapping_old_recalc(is_profile_run: bool, slot_mapping: List[List[int]],
                         seq_id: int, seq_len: int, context_len: int,
                         start_idx: int, block_size: int,
                         block_tables: Dict[int, List[int]],
                         num_layers: int = None,
                         prefill_config: Dict[str, Any] = None,
                         gen_config: Dict[str, Any] = None,
                         past_kv_seq_lens: List[int] = None,
                         recent_attn_weights_last_shape_list: List[int] = None):
    """
    Compute slot mapping.
    """
    # print(f"Compute slot mapping for number of layers: {num_layers}, seq_len: {seq_len}, context len: {context_len}")
    if is_profile_run:
        # During memory profiling, the block tables are not
        # initialized yet. In this case, we just use a dummy
        # slot mapping.
        # In embeddings, the block tables are {seq_id: None}.
        # slot_mapping.extend([PAD_SLOT_ID] * seq_len)
        # NOTE: not sure if this is correct 
        # Populate PAD_SLOT_ID for the num_layers        
        if len(slot_mapping) == 0:
            element = [PAD_SLOT_ID] * seq_len
            slot_mapping = [element for _ in range(num_layers)]
        else: 
            slot_mapping[0] = slot_mapping[0] + [PAD_SLOT_ID] * seq_len
            slot_mapping = [slot_mapping[0] for _ in range(num_layers)]
        
        # print(f"Shape of slot_mapping after profile run: {len(slot_mapping)}")
        return slot_mapping, past_kv_seq_lens, recent_attn_weights_last_shape_list

    # Mask the [0, start_idx) tokens of the prompt with
    # PAD_SLOT_ID, where start_idx is max(0, seq_len -
    # sliding_window). For example, if the prompt len is 10,
    # sliding window is 8, and block size is 4, the first two
    # tokens are masked and the slot mapping will be
    # [-1, -1, 2, 3, 4, 5, 6, 7, 0, 1].
    padding_mask_len = max(0, start_idx - context_len)
    # HACK: This is a temporary fix for the issue where the
    slot_mapping.extend([PAD_SLOT_ID] * padding_mask_len)


    block_table = block_tables[seq_id]
    is_decode_stage = int(context_len) != 0
    
    # NOTE: original 
    range_start = max(start_idx, context_len)
    range_end = seq_len
    numel = range_end - range_start

    # NOTE: random drop logic for decode stage 
    # Select a random position to place new token (drop the old token)
    
    # numpy implementation will be faster than python if we have
    # many elements, otherwise it will be slower.
    if numel < _COMPUTE_SLOT_MAPPING_NUMPY_NUMEL:
        _compute_slot_mapping_python(slot_mapping, block_table, range_start,
                                     range_end, block_size)
    else:
        _compute_slot_mapping_numpy(slot_mapping, block_table, range_start,
                                    range_end, block_size)
    
    slot_mappings = [copy.deepcopy(slot_mapping) for _ in range(num_layers)]

    if prefill_config is not None and gen_config is not None: 
        pass 
        # print("*"*25 + "Start Compute Slot Mapping" + "*"*25)
        # print(f"Slot mapping length: {len(slot_mapping)}, Context length: {context_len}, Sequence length: {seq_len}")
        # print(f"Start index: {start_idx}, range start: {range_start}, range_end: {range_end}")
        # if is_decode_stage:
        #     print(f"Seq id: {seq_id}. DECODE STAGE NOW")
        #     print(f"Slot mapping looks like this: {slot_mapping}")
        # else:
        #     print(f"Seq id: {seq_id}. PREFILL STAGE NOW")
        #     print(f"Slot mapping looks like: {slot_mapping}")
        # print("-->" + "Random Drop")  
    else:
        return slot_mappings
    
    # Random Drop, initialize parameter from config 
    prefill_decay_strategy = prefill_config["prefill_decay_strategy"]
    prefill_decay_ratio = prefill_config["prefill_decay_ratio"]
    min_context_length = prefill_config["min_context_length"]
    layerwise_downsample_interval = prefill_config["layerwise_downsample_interval"]
    recent_ratio = prefill_config["recent_ratio"]
    
    gen_decay_strategy = gen_config["gen_decay_strategy"]
    gen_decay_ratio = gen_config["gen_decay_ratio"]
    exceed_length_to_compress = gen_config["exceed_length_to_compress"]
    gen_compress_ratio = gen_config["gen_compress_ratio"]
    
    # Calculate a bit more parameters     
    inputs_embed_shape_1 = len(slot_mapping)
    
    if recent_attn_weights_last_shape_list is None:
        seq_length_with_past = inputs_embed_shape_1 + 0
    else:
        seq_length_with_past = inputs_embed_shape_1 + recent_attn_weights_last_shape_list[0]

    seq_length_with_past = seq_len
    recent_length = int(seq_length_with_past * recent_ratio)
    next_decoder_cache_seen_tokens = seq_len 
    # print(f"Recent length: {recent_length} (seq_length_with_past: {seq_length_with_past} * recent_ratio: {recent_ratio})")
    past_prefill_length = None 
    past_top_k = None
        
    if not is_decode_stage:
        # Prefill 
        assert past_kv_seq_lens is None, f"Past kv seq lens should be None for prefill stage, but got {past_kv_seq_lens}"
        assert recent_attn_weights_last_shape_list is None, f"Recent attn weights last shape list should be None for prefill stage, but got {recent_attn_weights_last_shape_list}"
        past_kv_seq_lens = []
        recent_attn_weights_last_shape_list = []
    else:
        # Decode 
        assert past_kv_seq_lens is not None and recent_attn_weights_last_shape_list is not None, f"Past kv seq lens should not be None for decode stage, but got {past_kv_seq_lens} and {recent_attn_weights_last_shape_list}"
        assert len(past_kv_seq_lens) == num_layers, f"Past kv seq lens length: {len(past_kv_seq_lens)} ({past_kv_seq_lens}) does not match num_layers: {num_layers}"
        assert len(recent_attn_weights_last_shape_list) == num_layers, f"Recent attn weights last shape list length: {len(recent_attn_weights_last_shape_list)} does not match num_layers: {num_layers}"
        
    for layer_id in range(num_layers):
        
        ### Prefill Stage ### 
        if not is_decode_stage:
            # determine the decay ratio schedule 
            
            if prefill_decay_strategy == "linear":
                schedule_prefill_decay_ratio = (1.0 - prefill_decay_ratio) * (layer_id / num_layers) + prefill_decay_ratio
            if prefill_decay_strategy == "cosine":
                schedule_prefill_decay_ratio = (1.0 - prefill_decay_ratio) * (math.cos(math.pi * layer_id / num_layers) + 1) / 2 + prefill_decay_ratio
            else:
                schedule_prefill_decay_ratio = prefill_decay_ratio
            
            num_tokens_candidate = seq_len - recent_length - 1
            prefill_context_len = past_prefill_length if past_prefill_length is not None else num_tokens_candidate
            
            # print(f"Past prefill length: {past_prefill_length}, prefill context length: {prefill_context_len}") 
            topk = None 
            # SELECT TOP K Tokens to keep
            if (layer_id % layerwise_downsample_interval) == 0:
                
                # print(f"Prefill stage, layer: {layer_id}, prefill context length: {prefill_context_len}, min context length: {min_context_length}, schedule prefill decay ratio: {schedule_prefill_decay_ratio}")
                if prefill_context_len > min_context_length and schedule_prefill_decay_ratio < 1.0: 
                    # print(f"layer id: {layer_id}, prefill context: {prefill_context_len}, schedule prefill decay ratio: {schedule_prefill_decay_ratio}, min context length: {min_context_length}")
                    topk = int(prefill_context_len * schedule_prefill_decay_ratio) if int(prefill_context_len * schedule_prefill_decay_ratio) > min_context_length else prefill_context_len
                    seq_len_to_keep = topk + recent_length + 1
                    # print(f"Layer: {layer_id}, Prefilling stage, k = {topk}, seq_len_to_keep: {seq_len_to_keep}")
                    
                    start_i = len(slot_mappings[layer_id]) - seq_len
                    end_i = start_i + seq_len
                    keep_end = start_i + seq_len_to_keep
                    assert keep_end <= len(slot_mappings[layer_id]), "Invalid range for seq_len_to_keep"

                    # NOTE: keep the last recent_length tokens 
                    replace_indices = list(range(start_i, keep_end - recent_length))
                    random.shuffle(replace_indices)

                    # Group replacement: replace indices beyond seq_len_to_keep
                    # TODO: does this mean I have to preserve what is thrown away in prev layer
                    dropping_tokens = 0
                    for origin_idx, replace_idx in zip(range(keep_end, end_i), replace_indices):                        
                        slot_mappings[layer_id][origin_idx] = slot_mappings[layer_id][replace_idx]
                        slot_mappings[layer_id][replace_idx] = PAD_SLOT_ID
                        dropping_tokens += 1
                        
                    # print(f"PREFILL STAGE - tokens dropped: {dropping_tokens}, slot mapping after random drop: {slot_mappings[layer_id]}")
                    past_prefill_length = seq_len_to_keep - recent_length - 1
                else:
                    # print(f"layer id: {layer_id}, prefill context: {prefill_context_len}, schedule prefill decay ratio: {schedule_prefill_decay_ratio}, min context length: {min_context_length}")
                    
                    # print(f"Layer: {layer_id}, Prefilling stage, k = {num_tokens_candidate}, keep all")
                    past_prefill_length = num_tokens_candidate
            else:
                # print(f"Layer: {layer_id}, Prefilling stage, k = {num_tokens_candidate}, keep all")
                past_prefill_length = num_tokens_candidate
                
            # Update past kv seq lens
            if past_top_k is None: 
                past_kv_seq_lens.append(num_tokens_candidate + recent_length + 1) 
            else:
                past_kv_seq_lens.append(past_top_k + recent_length + 1)
                
            # print(f"APPEND (past_kv_seq_lens): {past_kv_seq_lens}")
            # Update past top k if top k is not None
            past_top_k = topk if topk is not None else num_tokens_candidate
            
            # Set recent attn weights last shape list to the same as past kv seq lens
            recent_attn_weights_last_shape_list = [i for i in past_kv_seq_lens]
        ### Decode Stage ###
        else:
            # print(f"Decode stage, Layer: {layer_id}, past KV sequence lengths: {past_kv_seq_lens}")
            # determine the decay ratio schedule 
            if gen_decay_strategy == "linear":
                schedule_gen_decay_ratio = (1.0 - gen_decay_ratio) * (layer_id / num_layers) + gen_decay_ratio
            if gen_decay_strategy == "cosine":
                schedule_gen_decay_ratio = (1.0 - gen_decay_ratio) * (math.cos(math.pi * layer_id / num_layers) + 1) / 2 + gen_decay_ratio
            else:
                schedule_gen_decay_ratio = gen_decay_ratio
            
            if recent_attn_weights_last_shape_list is not None and past_kv_seq_lens is not None:
                predicted_top_k = clean_func(
                    recent_attn_weights_last_shape_list[layer_id],
                    past_kv_seq_lens[layer_id],
                    next_decoder_cache_seen_tokens, 
                    schedule_gen_decay_ratio, 
                    exceed_length_to_compress, 
                    recent_length, 
                    gen_compress_ratio,
                )            

                # print(f"Layer: {layer_id}, predicted top k: {predicted_top_k}")
                # NOTE: pyramidinfer configurations 
                # past_kv_seq_len = past_kv_seq_lens[layer_id]
                
                # TODO: SAMPLE and update slot mapping  
                # if current_kv_seq_len - recent_length - past_kv_seq_len >= exceed_length_to_compress:    
                #     sample_lists = []
                #     if seq_len_to_keep < block_size:
                #         # Case where seq_len is less than block_size
                #         # Add until seq_len_to_keep
                #         sample_lists = [block_table[0] * block_size + i for i in range(seq_len_to_keep)]
                #     else:
                #         max_blocks = seq_len_to_keep // block_size
                #         remainder = seq_len_to_keep % block_size
                #         print(f"Max number of blocks: {max_blocks}, remainder: {remainder}")
                #         for i in range(min(len(block_table), max_blocks)):
                #             # Add full block ranges
                #             sample_lists.extend([block_table[i] * block_size + j for j in range(block_size)])
                        
                #         if max_blocks < len(block_table) and remainder > 0:
                #             # Add the range for the remainder in the last block
                #             sample_lists.extend([block_table[max_blocks] * block_size + j for j in range(remainder)])
                    
                #     print(f"DECODE STAGE - sample lists: {sample_lists}")
                #     random_idx = random.choice(sample_lists) 
                #     print(f"replace {slot_mapping[-1]} with index {random_idx}")
                #     slot_mapping[-1] = random_idx
                # print(f"Slot mapping after decode stage modification: {slot_mapping}")
                
                # NOTE: update relevant parameters if top k is not None
                if predicted_top_k is not None:
                    old_value_attn = recent_attn_weights_last_shape_list[layer_id]
                    old_value_kv = past_kv_seq_lens[layer_id]
                    
                    # Update recent attn weights shape 
                    recent_attn_weights_last_shape_list[layer_id] = update_parameter(recent_attn_weights_last_shape_list, layer_id, recent_length, exceed_length_to_compress, predicted_top_k)
                    
                    # Update past KV seq lengths
                    past_kv_seq_lens[layer_id] = recent_attn_weights_last_shape_list[layer_id] - recent_length
                    
                    # print(f"[UTILS] layer id: {layer_id}, old value attn: {old_value_attn}, new value attn: {recent_attn_weights_last_shape_list[layer_id]}, old value kv: {old_value_kv}, new value kv: {past_kv_seq_lens[layer_id]}")
                else:
                    if recent_attn_weights_last_shape_list is not None:
                        recent_attn_weights_last_shape_list[layer_id] = recent_attn_weights_last_shape_list[layer_id] + 1  

    # slot_mapping is a list, repeat this list for num_layer times 
    assert len(slot_mappings) == num_layers, f"Slot mapping length: {len(slot_mappings)} does not match\
        num_layers: {num_layers}"
        
    if recent_attn_weights_last_shape_list is None:
        assert past_kv_seq_lens is not None, f"Past kv seq lens should not be None for decode stage, but got {past_kv_seq_lens}"
        recent_attn_weights_last_shape_list = past_kv_seq_lens 
        assert not is_decode_stage, f"Recent attn weights last shape list should be None for prefill stage, but got {recent_attn_weights_last_shape_list}"
    
    # print(f"Update slot mappings: {slot_mappings}, past KV seq lens: {past_kv_seq_lens}, recent attn weights last shape list: {recent_attn_weights_last_shape_list}")
    return slot_mappings, past_kv_seq_lens, recent_attn_weights_last_shape_list


TAttentionMetadata = TypeVar("TAttentionMetadata", bound='AttentionMetadata')


class CommonMetadataBuilder(AttentionMetadataBuilder[TAttentionMetadata]):

    _metadata_cls: Type[TAttentionMetadata]

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []
        self.num_prefills = 0
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0

        self.input_builder = input_builder
        self.runner = input_builder.runner

        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.use_v2_block_manager = (
            input_builder.scheduler_config.use_v2_block_manager)

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool):
        is_prompt = inter_data.is_prompt
        block_tables = inter_data.block_tables
        computed_block_nums = inter_data.computed_block_nums

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks):
            self.context_lens.append(context_len)
            if is_prompt:
                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
            else:
                assert query_len == 1, (
                    "seq_len: {}, context_len: {}, query_len: {}".format(
                        seq_len, context_len, query_len))
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if inter_data.prefix_cache_hit:
                block_table = computed_block_nums
            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                block_table = block_tables[seq_id][-curr_sliding_window_block:]
            self.block_tables.append(block_table)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(
                is_prompt, query_len, context_len, self.sliding_window,
                self.use_v2_block_manager)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)
        num_decode_tokens = self.num_decode_tokens

        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size

            # The shape of graph_block_tables is
            # [max batch size, max context len // block size].
            input_block_tables = self.runner.graph_block_tables[:batch_size]
            for i, block_table in enumerate(self.block_tables):
                if block_table:
                    input_block_tables[i, :len(block_table)] = block_table
            block_tables = torch.from_numpy(input_block_tables).to(
                device, non_blocking=True)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, "query_lens: {}".format(query_lens)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        query_lens_tensor = async_tensor_h2d(query_lens, torch.long, device,
                                             self.runner.pin_memory)  
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
                                            
        query_start_loc = torch.zeros(query_lens_tensor.shape[0] + 1,
                                      dtype=torch.int32,
                                      device=device)
        seq_start_loc = torch.zeros(seq_lens_tensor.shape[0] + 1,
                                    dtype=torch.int32,
                                    device=device)
        torch.cumsum(seq_lens_tensor,
                     dim=0,
                     dtype=seq_start_loc.dtype,
                     out=seq_start_loc[1:])
        torch.cumsum(query_lens_tensor,
                     dim=0,
                     dtype=query_start_loc.dtype,
                     out=query_start_loc[1:])

        return self._metadata_cls(  # type: ignore
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
        )


class CommonAttentionState(AttentionState):

    def __init__(self, runner: "ModelRunnerBase"):
        self.runner = runner
        self._is_graph_capturing = False

    @contextmanager
    def graph_capture(self, max_batch_size: int):
        self._is_graph_capturing = True
        self._graph_slot_mapping = torch.full((max_batch_size, ),
                                              PAD_SLOT_ID,
                                              dtype=torch.long,
                                              device=self.runner.device)
        self._graph_seq_lens = torch.ones(max_batch_size,
                                          dtype=torch.int32,
                                          device=self.runner.device)
        self._graph_block_tables = torch.from_numpy(
            self.runner.graph_block_tables).to(device=self.runner.device)
        yield
        self._is_graph_capturing = False
        del self._graph_slot_mapping
        del self._graph_seq_lens
        del self._graph_block_tables

    def graph_clone(self, batch_size: int) -> "CommonAttentionState":
        assert self._is_graph_capturing
        return self.__class__(self.runner)

    def graph_capture_get_metadata_for_batch(
            self, batch_size: int, is_encoder_decoder_model: bool = False):
        assert self._is_graph_capturing
        attn_metadata = self.runner.attn_backend.make_metadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=batch_size,
            slot_mapping=self._graph_slot_mapping[:batch_size],
            seq_lens=None,
            seq_lens_tensor=self._graph_seq_lens[:batch_size],
            max_query_len=1,
            max_decode_query_len=1,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.runner.max_seq_len_to_capture,
            query_start_loc=None,
            seq_start_loc=None,
            context_lens_tensor=None,
            block_tables=self._graph_block_tables[:batch_size],
            use_cuda_graph=True,
            
        )
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers backend.
            # Assert the same.
            assert self.runner.attn_backend.get_name() == "xformers", \
            f"Expected attn_backend name to be 'xformers', but "\
            f" got '{self.runner.attn_backend.get_name()}'"
            self._update_captured_metadata_for_enc_dec_model(
                batch_size=batch_size, attn_metadata=attn_metadata)

        return attn_metadata

    def get_graph_input_buffers(
            self,
            attn_metadata,
            is_encoder_decoder_model: bool = False) -> Dict[str, Any]:
        input_buffers = {
            "slot_mapping": attn_metadata.slot_mapping,
            "seq_lens_tensor": attn_metadata.decode_metadata.seq_lens_tensor,
            "block_tables": attn_metadata.decode_metadata.block_tables,
        }
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers backend.
            # Assert the same.
            assert self.runner.attn_backend.get_name() == "xformers", \
            f"Expected attn_backend name to be 'xformers', but "\
            f" got '{self.runner.attn_backend.get_name()}'"
            self._add_additonal_input_buffers_for_enc_dec_model(
                attn_metadata=attn_metadata, input_buffers=input_buffers)
        return input_buffers

    def prepare_graph_input_buffers(
            self,
            input_buffers,
            attn_metadata,
            is_encoder_decoder_model: bool = False) -> None:
        input_buffers["seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.seq_lens_tensor, non_blocking=True)
        input_buffers["block_tables"].copy_(
            attn_metadata.decode_metadata.block_tables, non_blocking=True)
        if is_encoder_decoder_model:
            # The encoder decoder model works only with XFormers backend.
            # Assert the same.
            assert self.runner.attn_backend.get_name() == "xformers", \
            f"Expected attn_backend name to be 'xformers', but "\
            f" got '{self.runner.attn_backend.get_name()}'"
            self._prepare_input_buffers_for_enc_dec_model(
                attn_metadata, input_buffers)

    def begin_forward(self, model_input) -> None:
        return

    def _update_captured_metadata_for_enc_dec_model(self, batch_size: int,
                                                    attn_metadata):
        """
        Updates the attention metadata parameters for CUDA graph capture in an
        encoder-decoder model.

        This method modifies attention-related tensors and metadata required
        for CUDA graph capture in encoder-decoder models. Specifically, it
        updates the cross-attention and encoder sequence tensors in the 
        AttentionMetadata object.
        """
        # During decode phase the cross_slot_mapping will be empty. Hence set
        # an empty tensor for CUDA Graph capture.
        attn_metadata.cross_slot_mapping = torch.tensor(
            [], dtype=torch.int).cuda()
        attn_metadata.cross_block_tables = torch.full(
            (batch_size, self.runner.get_max_block_per_batch()),
            1,
            dtype=torch.int).cuda()
        attn_metadata.encoder_seq_lens = torch.full((batch_size, ),
                                                    1,
                                                    dtype=torch.int).cuda()
        attn_metadata.encoder_seq_lens_tensor = torch.full(
            (batch_size, ), 1, dtype=torch.int).cuda()
        attn_metadata.max_encoder_seq_len = self.runner.max_seq_len_to_capture

    def _add_additonal_input_buffers_for_enc_dec_model(
            self, attn_metadata, input_buffers: Dict[str, Any]):
        """
        Saves additional input buffers specific to the encoder-decoder model
        from the attention metadata.

        This method extracts and stores encoder-decoder related input buffers
        from the `attn_metadata` into the `input_buffers` dictionary. The
        buffers include encoder sequence lengths, cross-slot mappings, and
        cross-block tables, which are essential for the encoder-decoder model
        during CUDA graph replay.
        """
        input_buffers["encoder_seq_lens_tensor"] = (
            attn_metadata.decode_metadata.encoder_seq_lens_tensor)
        input_buffers["cross_slot_mapping"] = (
            attn_metadata.decode_metadata.cross_slot_mapping)
        input_buffers["cross_block_tables"] = (
            attn_metadata.decode_metadata.cross_block_tables)

    def _prepare_input_buffers_for_enc_dec_model(self, attn_metadata,
                                                 input_buffers: Dict[str,
                                                                     Any]):
        """
        Populates input buffers with data from the encoder-decoder model's
        attention metadata.

        This method fills the input buffers with encoder-decoder specific
        tensors. It copies data from the `attn_metadata` and keyword arguments
        (`kwargs`) into corresponding buffers in the `input_buffers` dictionary.
        The copied data includes attention-related metadata as well as input 
        IDs and positional information for the encoder.
        """
        input_buffers["encoder_seq_lens_tensor"].copy_(
            attn_metadata.decode_metadata.encoder_seq_lens_tensor,
            non_blocking=True)
        input_buffers["cross_slot_mapping"].copy_(
            attn_metadata.decode_metadata.cross_slot_mapping,
            non_blocking=True)
        input_buffers["cross_block_tables"].copy_(
            attn_metadata.decode_metadata.cross_block_tables,
            non_blocking=True)
