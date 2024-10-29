# Import methods in this file to configure per-layer block table
from abc import abstractmethod
import math
from typing import List, Protocol, Dict, Any, Tuple, Type, TYPE_CHECKING
import torch
from typing_extensions import TypeVar
from vllm.config import KVCacheConfig, ModelConfig, ParallelConfig
from vllm.core.block.block_table import BlockTable
from vllm.core.block.interfaces import Block, DeviceAwareBlockAllocator
from vllm.core.block.prefix_caching_block import ComputedBlocksTracker, LastAccessBlocksTracker
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import STR_DTYPE_TO_TORCH_DTYPE, Device, cdiv, chunk_list, get_dtype_size
from vllm.logger import init_logger


class AppAwareManager:

    def __init__(self, dtype: torch.dtype):
        self.initialized = False
        assert isinstance(dtype, torch.dtype)
        self.dtype = dtype

    def init_kv_cache_config(self, kv_cache_config: KVCacheConfig):
        assert not self.initialized, "KV cache config is already initialized"
        # self.kv_cache_config = kv_cache_config
        # TODO: can we remove this function?
        self.initialized = True

    @abstractmethod
    def get_page_size(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_app_property(self) -> str:
        # app with the same property can share the same block table
        raise NotImplementedError

    @abstractmethod
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        # FIXME(heheda12345): When implementing this interface,  we assume that
        # all sequences in the group share the same prompt. This is the same as
        # BlockSpaceManagerV2.
        raise NotImplementedError

    @abstractmethod
    def allocate_sequence(self, seq_group: SequenceGroup,
                          block_allocator: DeviceAwareBlockAllocator,
                          group_id: str) -> BlockTable:
        raise NotImplementedError

    @abstractmethod
    def get_possible_hit_lens(
            self, block_is_computed: List[bool]) -> List[Tuple[int, int]]:
        raise NotImplementedError

    @abstractmethod
    def get_prefix_cache_alignment(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def filter_computed_blocks_by_token(
            self, computed_tokens: int, block_table: BlockTable,
            block_allocator: DeviceAwareBlockAllocator):
        raise NotImplementedError

    @abstractmethod
    def update_seq_blocks_last_access(
            self, seq: Sequence, block_table: BlockTable,
            last_access_blocks_tracker: LastAccessBlocksTracker):
        raise NotImplementedError


AppAwareAttnMetadataBuilder = AppAwareManager


def get_token_size_default(model_config: ModelConfig,
                           parallel_config: ParallelConfig,
                           dtype: torch.dtype) -> int:
    head_size = model_config.get_head_size()
    num_heads = model_config.get_num_kv_heads(parallel_config)
    num_attention_layers = 1

    key_cache_block = num_heads * head_size
    value_cache_block = key_cache_block
    total = num_attention_layers * (key_cache_block + value_cache_block)
    # dtype_size = get_dtype_size(dtype)
    return total


def get_dtype(cache_dtype: str, model_config: ModelConfig) -> torch.dtype:
    if cache_dtype == "auto":
        return model_config.dtype
    return STR_DTYPE_TO_TORCH_DTYPE[cache_dtype]


def suffix_to_mutable_block(block_table: BlockTable,
                            block_allocator: DeviceAwareBlockAllocator,
                            from_idx: int):
    next_block = None
    for idx, block in zip(
            range(len(block_table._blocks) - 1, from_idx - 1, -1),
            block_table._blocks[:from_idx - 1:-1]):
        if block.computed:
            device = Device.GPU
            block_allocator.free(block)
            new_block = block_allocator.allocate_mutable_block(
                block._prev_block, device, block._group_id_hash)
            if next_block is not None:
                next_block._prev_block = new_block
            next_block = new_block
            block_table._blocks[idx] = new_block
        else:
            next_block = block


def prefix_to_null_block(block_table: BlockTable,
                         block_allocator: DeviceAwareBlockAllocator,
                         num_blocks: int):
    if num_blocks <= 0:
        return
    null_block = block_allocator.allocate_or_get_null_block()
    for idx, block in enumerate(block_table._blocks[:num_blocks]):
        if block is not null_block:
            block_allocator.free(block)
            block_table._blocks[idx] = null_block
    if num_blocks < len(block_table._blocks):
        block_table._blocks[num_blocks]._prev_block = null_block
        # do not need to compute hash based on prev_blocks
        assert block_table._blocks[num_blocks]._cached_content_hash is not None


class SelfAttentionManager(AppAwareManager):

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_dtype: str,
                 block_size: int):
        super().__init__(get_dtype(cache_dtype, model_config))
        self.memory_per_token = get_token_size_default(model_config,
                                                       parallel_config,
                                                       self.dtype)
        self.block_size = block_size

    def get_page_size(self):
        return self.block_size * self.memory_per_token

    def get_app_property(self) -> str:
        return "self_attention"

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_tokens = len(seq.get_token_ids())
        return cdiv(num_tokens, self.block_size) + num_lookahead_slots

    def allocate_sequence(self, seq_group: SequenceGroup,
                          block_allocator: DeviceAwareBlockAllocator,
                          group_id: str) -> BlockTable:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        block_table = BlockTable(block_size=self.block_size,
                                 block_allocator=block_allocator,
                                 max_block_sliding_window=None,
                                 group_id=group_id,
                                 seq_id=seq.seq_id)
        block_table.allocate(seq.get_token_ids())

        return block_table

    def get_num_blocks_touched_by_append_slots(self, seq: Sequence,
                                               block_table: BlockTable,
                                               num_lookahead_slots: int):
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())

        num_token_ids = len(unseen_token_ids) + num_lookahead_slots
        first_chunk_size = self.block_size - (block_table._num_full_slots %
                                              self.block_size)
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self.block_size))
        return num_token_blocks

    def append_token_ids(self, seq: Sequence, block_table: BlockTable,
                         num_lookahead_slots: int,
                         last_access_blocks_tracker: LastAccessBlocksTracker):
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())

        block_table.ensure_num_empty_slots(
            num_empty_slots=len(unseen_token_ids) + num_lookahead_slots)

        # Update the blocks with the new tokens
        first_block_idx = block_table._num_full_slots // self.block_size
        token_blocks = block_table._chunk_token_blocks_for_append(
            unseen_token_ids)

        for i, token_block in enumerate(token_blocks):
            block_table._blocks.append_token_ids(first_block_idx + i,
                                                 token_block)

        block_table._num_full_slots += len(unseen_token_ids)

    def get_prefix_cache_alignment(self) -> int:
        return self.block_size

    def get_possible_hit_lens(
            self, block_is_computed: List[bool]) -> List[Tuple[int, int]]:
        assert block_is_computed[-1] is False
        hit_until = block_is_computed.index(False)
        return [(0, hit_until * self.block_size)]

    def filter_computed_blocks_by_token(
            self, computed_tokens: int, block_table: BlockTable,
            block_allocator: DeviceAwareBlockAllocator):
        assert computed_tokens % self.block_size == 0
        num_blocks = computed_tokens // self.block_size
        suffix_to_mutable_block(block_table, block_allocator, num_blocks)
        return block_table.physical_block_ids[:num_blocks]

    def update_seq_blocks_last_access(
            self, seq: Sequence, block_table: BlockTable,
            last_access_blocks_tracker: LastAccessBlocksTracker):
        last_access_blocks_tracker.update_seq_blocks_last_access(
            seq.seq_id, block_table.physical_block_ids)


class EncoderDecoderManager(AppAwareManager):

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_dtype: str,
                 block_size: int):
        super().__init__(get_dtype(cache_dtype, model_config))
        self.memory_per_token = get_token_size_default(model_config,
                                                       parallel_config,
                                                       self.dtype)
        self.block_size = block_size

    def get_page_size(self):
        return self.block_size * self.memory_per_token

    def get_app_property(self) -> str:
        return "encoder_decoder"

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                _num_lookahead_slots: int = 0) -> int:
        if seq_group.is_encoder_decoder():
            encoder_seq = seq_group.get_encoder_seq()
            assert encoder_seq is not None
            num_tokens = len(encoder_seq.get_token_ids())
            return cdiv(num_tokens, self.block_size)
        else:
            return 0

    def allocate_sequence(self, seq_group: SequenceGroup,
                          block_allocator: DeviceAwareBlockAllocator,
                          group_id: str) -> BlockTable:
        encoder_seq = seq_group.get_encoder_seq()
        block_table = BlockTable(block_size=self.block_size,
                                 block_allocator=block_allocator,
                                 max_block_sliding_window=None,
                                 group_id=group_id,
                                 seq_id=encoder_seq.seq_id)
        encoder_seq_token_ids = encoder_seq.get_token_ids()
        if encoder_seq_token_ids:
            block_table.allocate(encoder_seq_token_ids)
        return block_table

    def get_num_blocks_touched_by_append_slots(self, seq, block_table,
                                               num_lookahead_slots):
        # Encoder-decoder KV cache size is not changed during decoding
        return 0

    def append_token_ids(self, seq, block_table, num_lookahead_slots,
                         last_access_blocks_tracker: LastAccessBlocksTracker):
        # Encoder-decoder KV cache size is not changed during decoding
        pass

    def get_prefix_cache_alignment(self) -> int:
        return self.block_size


class SlidingWindowManager(AppAwareManager):

    def __init__(self, model_config: ModelConfig,
                 parallel_config: ParallelConfig, cache_dtype: str,
                 block_size: int, sliding_window_size: int):
        super().__init__(get_dtype(cache_dtype, model_config))
        self.memory_per_token = get_token_size_default(model_config,
                                                       parallel_config,
                                                       self.dtype)
        assert sliding_window_size > 0
        self.sliding_window_size = sliding_window_size
        self.block_size = block_size
        # +1 here because // rounds down
        num_blocks = (self.sliding_window_size - 1) // self.block_size + 1
        # +1 here because the last block may not be full,
        # and so the sequence stretches one more block at the beginning
        # For example, if sliding_window is 3 and block_size is 4,
        # we may need 2 blocks when the second block only holds 1 token.
        self.max_block_sliding_window = num_blocks + 1
        print("max_block_sliding_window", self.max_block_sliding_window)

    def get_page_size(self):
        return self.block_size * self.memory_per_token

    def get_app_property(self) -> str:
        return f"sliding_window_{self.sliding_window_size}"

    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        num_tokens = len(seq.get_token_ids())
        num_required_blocks = cdiv(num_tokens + num_lookahead_slots,
                                   self.block_size)  # ????
        # Do not calculate min here, as prefill phase allocates all blocks
        # num_required_blocks = min(num_required_blocks,
        #                           self.max_block_sliding_window)
        return num_required_blocks

    def allocate_sequence(self, seq_group: SequenceGroup,
                          block_allocator: DeviceAwareBlockAllocator,
                          group_id: str) -> BlockTable:
        seq = seq_group.get_seqs(status=SequenceStatus.WAITING)[0]
        # TODO: handle sliding window in this manager, so that we can remove the
        # sliding window logic in other parts of the code
        block_table = BlockTable(
            block_size=self.block_size,
            block_allocator=block_allocator,
            max_block_sliding_window=self.max_block_sliding_window,
            group_id=group_id,
            seq_id=seq.seq_id)
        block_table.allocate(seq.get_token_ids())

        return block_table

    def get_num_blocks_touched_by_append_slots(self, seq: Sequence,
                                               block_table: BlockTable,
                                               num_lookahead_slots: int):
        # TODO: this function may add redundant blocks to the block table
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())

        num_token_ids = len(unseen_token_ids) + num_lookahead_slots
        first_chunk_size = self.block_size - (block_table._num_full_slots %
                                              self.block_size)
        num_token_blocks = (1 + math.ceil(
            (num_token_ids - first_chunk_size) / self.block_size))
        return num_token_blocks

    def append_token_ids(self, seq: Sequence, block_table: BlockTable,
                         num_lookahead_slots: int,
                         last_access_blocks_tracker: LastAccessBlocksTracker):
        assert block_table._block_size == self.block_size
        unseen_token_ids = block_table.get_unseen_token_ids(
            seq.get_token_ids())
        num_computed_slots = seq.data.get_num_computed_tokens()

        null_block = block_table._allocator.allocate_or_get_null_block()
        assert num_computed_slots is not None
        end_block_idx = (num_computed_slots //
                         self.block_size) - self.max_block_sliding_window
        if block_table._allocator.allocator_type == "prefix_caching":
            last_access_blocks_tracker.update_seq_blocks_last_access(
                block_table._seq_id, [
                    b.block_id for b in block_table._blocks[:end_block_idx]
                    if b is not null_block
                ])
        for idx in range(0, end_block_idx):
            b = block_table._blocks[idx]
            if b is not null_block:
                block_table._allocator.free(b)
                block_table._blocks[idx] = null_block

        block_table.ensure_num_empty_slots(
            num_empty_slots=len(unseen_token_ids) + num_lookahead_slots)

        # Update the blocks with the new tokens
        first_block_idx = block_table._num_full_slots // self.block_size
        token_blocks = block_table._chunk_token_blocks_for_append(
            unseen_token_ids)

        for i, token_block in enumerate(token_blocks):
            block_table._blocks.append_token_ids(first_block_idx + i,
                                                 token_block)

        block_table._num_full_slots += len(unseen_token_ids)

    def get_prefix_cache_alignment(self) -> int:
        return self.block_size

    def get_possible_hit_lens(
            self, block_is_computed: List[bool]) -> List[Tuple[int, int]]:
        assert block_is_computed[-1] is False
        start = 0
        ranges = []
        for i, is_computed in enumerate(block_is_computed):
            if not is_computed:
                if start == 0:
                    ranges.append(
                        (start * self.block_size, i * self.block_size))
                elif i - start >= self.max_block_sliding_window:
                    ranges.append(((start + self.max_block_sliding_window) *
                                   self.block_size, i * self.block_size))
                start = i + 1
        return ranges

    def filter_computed_blocks_by_token(
            self, computed_tokens: int, block_table: BlockTable,
            block_allocator: DeviceAwareBlockAllocator):
        assert computed_tokens % self.block_size == 0
        num_blocks = computed_tokens // self.block_size

        suffix_to_mutable_block(block_table, block_allocator, num_blocks)
        # 1 is a magic number to make the touched range a little larger
        prefix_to_null_block(block_table, block_allocator,
                             num_blocks - self.max_block_sliding_window - 1)
        return block_table.physical_block_ids[:num_blocks]

    def update_seq_blocks_last_access(
            self, seq: Sequence, block_table: BlockTable,
            last_access_blocks_tracker: LastAccessBlocksTracker):
        last_access_blocks_tracker.update_seq_blocks_last_access(
            seq.seq_id, block_table.physical_block_ids)
