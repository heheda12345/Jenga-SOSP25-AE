from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

import msgspec

from vllm.config import ModelConfig
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.prefix_caching_block import ComputedBlocksTracker, LastAccessBlocksTracker
from vllm.core.block_v3.custom_block_manager import CustomBlockManager, CUSTOM_BLOCK_TABLE
from vllm.core.interfaces import AllocStatus, BlockSpaceManager, PER_LAYER_BLOCK_IDS
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.logger import init_logger

logger = init_logger(__name__)
SeqId = int


class PerlayerBlockSpaceManager(BlockSpaceManager):

    def __init__(
        self,
        block_size: int,
        custom_block_manager: CustomBlockManager,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.custom_block_manager = custom_block_manager
        self.enable_caching = enable_caching

        self.num_total_gpu_blocks = self.custom_block_manager.kv_cache_config.buffer_size // self.custom_block_manager.kv_cache_config.level0_page_size
        self.num_total_cpu_blocks = 0
        if sliding_window is not None:
            logger.warning("sliding_window can be deprecated in the future.")

        self.global_block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=self.num_total_gpu_blocks,
            num_cpu_blocks=self.num_total_cpu_blocks,
            block_size=block_size,
        )

        self.watermark_blocks = int(watermark * self.num_total_gpu_blocks)
        logger.info(
            "############### create PerlayerBlockSpaceManager, block_size: {}, page size: {}"
            .format(
                block_size,
                self.custom_block_manager.kv_cache_config.level0_page_size), )

        self.block_tables: Dict[SeqId, CUSTOM_BLOCK_TABLE] = {}

        self._computed_blocks_tracker = ComputedBlocksTracker(
            self.global_block_allocator)
        self._last_access_blocks_tracker = LastAccessBlocksTracker(
            self.global_block_allocator)

    def add_model(self, model: ModelConfig):
        self.custom_block_manager.add_block_managers_of_model(model)

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:

        num_required_blocks = self.custom_block_manager.get_num_required_blocks(
            seq_group,
            num_lookahead_slots=num_lookahead_slots,
        )

        num_free_gpu_blocks = self.global_block_allocator.get_num_free_blocks(
            device=Device.GPU)

        if (self.num_total_gpu_blocks - num_required_blocks <
                self.watermark_blocks):
            return AllocStatus.NEVER
        if num_free_gpu_blocks - num_required_blocks >= self.watermark_blocks:
            return AllocStatus.OK
        else:
            return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup) -> None:
        waiting_seqs = seq_group.get_seqs(status=SequenceStatus.WAITING)

        block_table: CUSTOM_BLOCK_TABLE = self.custom_block_manager \
            .allocate_sequence(seq_group, self.global_block_allocator)

        seq_id = seq_group.seqs[0].seq_id
        self.block_tables[seq_id] = block_table

        self._computed_blocks_tracker.add_seq(seq_id)
        self._last_access_blocks_tracker.add_seq(seq_id)

        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

            # Track seq
            self._computed_blocks_tracker.add_seq(seq.seq_id)
            self._last_access_blocks_tracker.add_seq(seq.seq_id)

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        num_touched_blocks = 0
        for seq in seq_group.get_seqs(status=SequenceStatus.RUNNING):
            block_table = self.block_tables[seq.seq_id]
            num_touched_blocks += self.custom_block_manager.get_num_blocks_touched_by_append_slots(
                seq, block_table, num_lookahead_slots)
        num_free_gpu_blocks = self.global_block_allocator.get_num_free_blocks(
            Device.GPU)
        return num_touched_blocks <= num_free_gpu_blocks

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        block_table = self.block_tables[seq.seq_id]
        self.custom_block_manager.append_token_ids(seq, block_table,
                                                   num_lookahead_slots)
        new_cows = self.global_block_allocator.clear_copy_on_writes()
        return new_cows

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: PerlayerBlockSpaceManager.fork")

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: PerlayerBlockSpaceManager.can_swap_in")

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: PerlayerBlockSpaceManager.swap_in")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: PerlayerBlockSpaceManager.can_swap_out")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: PerlayerBlockSpaceManager.swap_out")

    def free(self, seq: Sequence) -> None:
        seq_id = seq.seq_id

        if seq_id not in self.block_tables:
            # Already freed or haven't been scheduled yet.
            return

        # Update seq block ids with the latest access time
        self.custom_block_manager.update_seq_blocks_last_access(
            seq, self.block_tables[seq.seq_id],
            self._last_access_blocks_tracker)

        # Untrack seq
        self._last_access_blocks_tracker.remove_seq(seq_id)
        self._computed_blocks_tracker.remove_seq(seq_id)

        # Free table/blocks
        for block in self.block_tables[seq_id].values():
            block.free()
        del self.block_tables[seq_id]

    def get_block_table_for_exec(self, seq: Sequence) -> PER_LAYER_BLOCK_IDS:
        block_tables = self.block_tables[seq.seq_id]
        block_ids = {
            block_id: block_tables[block_id].physical_block_ids_for_exec
            for block_id in block_tables
        }
        return block_ids

    def get_num_free_gpu_blocks(self) -> int:
        return self.global_block_allocator.get_num_free_blocks(Device.GPU)

    def get_num_free_cpu_blocks(self) -> int:
        return self.global_block_allocator.get_num_free_blocks(Device.CPU)

    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.enable_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        assert len(seqs) == 1
        return self.custom_block_manager.get_common_computed_block_ids(
            seqs[0], self.block_tables[seqs[0].seq_id],
            self._computed_blocks_tracker)

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        # mark all touched blocks as computed
        self.global_block_allocator.mark_blocks_as_computed([])

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        return self.global_block_allocator.get_prefix_cache_hit_rate(
            Device.GPU)

    # for the compatibility with current Scheduler. Can be removed later
    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        return []

    # for the compatibility with current Scheduler. Can be removed later
    def free_cross(self, seq_group: SequenceGroup) -> None:
        pass
