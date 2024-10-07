from typing import Dict, List, Optional
from typing import Sequence as GenericSequence
from typing import Tuple

from vllm.config import ModelConfig
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block_v3.custom_block_manager import CustomBlockManager, CUSTOM_BLOCK_TABLE
from vllm.core.interfaces import AllocStatus, BlockSpaceManager, PER_LAYER_BLOCK_IDS
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus
from vllm.utils import Device
from vllm.logger import init_logger

logger = init_logger(__name__)
SeqId = int


class BlockSpaceManagerV3(BlockSpaceManager):

    def __init__(
        self,
        block_size: int,
        num_gpu_blocks: int,
        num_cpu_blocks: int,
        watermark: float = 0.01,
        sliding_window: Optional[int] = None,
        enable_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        self.num_total_gpu_blocks = num_gpu_blocks
        self.num_total_cpu_blocks = num_cpu_blocks
        if sliding_window is not None:
            raise NotImplementedError("Sliding window is not supported")
        if enable_caching:
            raise NotImplementedError("Prefix caching is not supported")

        self.global_block_allocator = CpuGpuBlockAllocator.create(
            allocator_type="prefix_caching" if enable_caching else "naive",
            num_gpu_blocks=num_gpu_blocks,
            num_cpu_blocks=num_cpu_blocks,
            block_size=block_size,
        )

        self.custom_block_manager = CustomBlockManager(
            block_size=block_size, block_allocator=self.global_block_allocator)

        self.enable_caching = enable_caching

        self.watermark_blocks = int(watermark * num_gpu_blocks)
        logger.info(
            "############### create BlockSpaceManagerV3, block_size: {}".
            format(block_size))

        self.block_tables: Dict[SeqId, CUSTOM_BLOCK_TABLE] = {}

    def add_model(self, model: ModelConfig):
        self.custom_block_manager.add_block_managers_of_model(model)

    def can_allocate(self,
                     seq_group: SequenceGroup,
                     num_lookahead_slots: int = 0) -> AllocStatus:

        num_required_blocks = self.custom_block_manager.get_num_required_blocks(
            seq_group,
            block_size=self.block_size,
            num_lookahead_slots=num_lookahead_slots,
        )
        print("num_required_blocks: ", num_required_blocks)

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
            .allocate_sequence(seq_group)
        self.block_tables[seq_group.seqs[0].seq_id] = block_table

        for seq in waiting_seqs[1:]:
            self.block_tables[seq.seq_id] = block_table.fork()

    def can_append_slots(self, seq_group: SequenceGroup,
                         num_lookahead_slots: int) -> bool:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.can_append_slots")

    def append_slots(
        self,
        seq: Sequence,
        num_lookahead_slots: int,
    ) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.append_slots")

    def fork(self, parent_seq: Sequence, child_seq: Sequence) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError("not implemented: BlockSpaceManagerV3.fork")

    def can_swap_in(self, seq_group: SequenceGroup,
                    num_lookahead_slots: int) -> AllocStatus:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.can_swap_in")

    def swap_in(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.swap_in")

    def can_swap_out(self, seq_group: SequenceGroup) -> bool:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.can_swap_out")

    def swap_out(self, seq_group: SequenceGroup) -> List[Tuple[int, int]]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.swap_out")

    def free(self, seq: Sequence) -> None:
        import pdb
        pdb.set_trace()
        raise NotImplementedError("not implemented: BlockSpaceManagerV3.free")

    def get_block_table(self, seq: Sequence) -> PER_LAYER_BLOCK_IDS:
        block_tables = self.block_tables[seq.seq_id]
        block_ids = {
            block_id: block_tables[block_id].physical_block_ids
            for block_id in block_tables
        }
        print("[[[block_ids: ]]]")
        for layer_id in block_ids:
            print(f"layer: {layer_id}, block_ids: {block_ids[layer_id]}")
            if layer_id == 3: break
        return block_ids

    def get_num_free_gpu_blocks(self) -> int:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_num_free_gpu_blocks")

    def get_num_free_cpu_blocks(self) -> int:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_num_free_cpu_blocks")

    def access_all_blocks_in_seq(
        self,
        seq: Sequence,
        access_time: float,
    ) -> None:
        if self.enable_caching:
            import pdb
            pdb.set_trace()
            raise NotImplementedError(
                "not implemented: BlockSpaceManagerV3.access_all_blocks_in_seq"
            )

    def get_common_computed_block_ids(
            self, seqs: List[Sequence]) -> GenericSequence[int]:
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_common_computed_block_ids"
        )

    def mark_blocks_as_computed(self, seq_group: SequenceGroup,
                                token_chunk_size: int):
        if self.enable_caching:
            import pdb
            pdb.set_trace()
            raise NotImplementedError(
                "not implemented: BlockSpaceManagerV3.mark_blocks_as_computed")

    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        """Prefix cache hit rate. -1 means not supported or disabled."""
        import pdb
        pdb.set_trace()
        raise NotImplementedError(
            "not implemented: BlockSpaceManagerV3.get_prefix_cache_hit_rate")

    # for the compatibility with current Scheduler. Can be removed later
    def get_cross_block_table(self, seq_group: SequenceGroup) -> List[int]:
        return []
