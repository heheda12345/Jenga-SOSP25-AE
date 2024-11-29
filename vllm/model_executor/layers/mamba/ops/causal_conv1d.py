# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py

from typing import Optional

import torch

from vllm import _custom_ops as ops
from vllm.attention.backends.utils import LAMetadata


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    query_start_loc: Optional[torch.Tensor] = None,
    cache_indices: Optional[torch.Tensor] = None,
    has_initial_state: Optional[torch.Tensor] = None,
    conv_states: Optional[torch.Tensor] = None,
    activation: Optional[str] = "silu",
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]), 
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index, 
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial 
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"

    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out = ops.causal_conv1d_fwd(x, weight, bias, conv_states, query_start_loc,
                                cache_indices, has_initial_state, activation
                                in ["silu", "swish"])
    return out


def causal_conv1d_fn_with_cache(x: torch.Tensor,
                                weight: torch.Tensor,
                                bias: Optional[torch.Tensor],
                                conv_cache: torch.Tensor,
                                conv_metadata: LAMetadata,
                                activation: Optional[str] = "silu"):
    """
    x: (cu_seq_len, dim), do transpose in this function and pass to ops as (dim, cu_seq_len)
    weight: (dim, width)
    bias: (dim,)
    return: (cu_seq_len, dim)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")

    bias = bias.contiguous() if bias is not None else None
    if conv_metadata.num_prefills == 0:
        # only decode, need to be cuda-graph capture-able
        x = x.transpose(-2, -1)
        if x.stride(-1) != 1:
            x = x.contiguous()
        decode_metadata = conv_metadata.steps[0]
        out = ops.causal_conv1d_fwd(x, weight, bias, conv_cache,
                                    decode_metadata.query_start_loc,
                                    decode_metadata.cache_indices,
                                    decode_metadata.context_lens_tensor > 0,
                                    activation in ["silu", "swish"])
        out = out.transpose(-2, -1)
        return out

    out_tensor = torch.empty_like(x)
    # mix of prefill and decode
    for step in conv_metadata.steps:
        x_step = x[step.token_positions].transpose(-2, -1).contiguous()
        conv_cache[step.curr_block_table] = conv_cache[step.prev_block_table]
        if not conv_metadata.is_profile_run:
            out = ops.causal_conv1d_fwd(x_step, weight, bias, conv_cache,
                                        step.query_start_loc,
                                        step.cache_indices,
                                        step.context_lens_tensor > 0,
                                        activation in ["silu", "swish"])
        else:
            out = torch.empty_like(x_step)
        out_tensor[step.token_positions] = out.transpose(-2, -1)
    # ensure the last block is mutable
    if conv_metadata.need_final_copy:
        conv_cache[conv_metadata.to_mutable_dst] = conv_cache[
            conv_metadata.to_mutable_src]

    return out_tensor


def causal_conv1d_update(x: torch.Tensor,
                         conv_state: torch.Tensor,
                         weight: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         activation: Optional[str] = None,
                         cache_seqlens: Optional[torch.Tensor] = None,
                         conv_state_indices: Optional[torch.Tensor] = None):
    """
    x: (batch, dim) or (batch, dim, seqlen)
    conv_state: (batch, dim, state_len), where state_len >= width - 1
    weight: (dim, width)
    bias: (dim,)
    cache_seqlens: (batch,), dtype int32.
        If not None, the conv_state is treated as a circular buffer.
        The conv_state will be updated by copying x to the conv_state 
        starting at the index
        @cache_seqlens % state_len.
    conv_state_indices: (batch,), dtype int32
        If not None, the conv_state is a larger tensor along the batch dim, 
        and we are selecting the batch coords specified by conv_state_indices.
        Useful for a continuous batching scenario.

    out: (batch, dim) or (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    activation_val = activation in ["silu", "swish"]
    unsqueeze = x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(-1)
    out = ops.causal_conv1d_update(x, conv_state, weight, bias, activation_val,
                                   cache_seqlens, conv_state_indices)
    if unsqueeze:
        out = out.squeeze(-1)
    return out
