from contextlib import nullcontext
from dataclasses import dataclass
import math
from typing import List, Optional, Dict, NewType, Tuple

from vllm.config import CacheConfig, ComponentType, KVCacheConfig, KVPageType, ModelConfig, ParallelConfig
from vllm.core.block.common import BlockList
from vllm.core.block.interfaces import (
    Block,
    DeviceAwareBlockAllocator,
)
from vllm.core.block.prefix_caching_block import ComputedBlocksTracker, LastAccessBlocksTracker
from vllm.core.interfaces import ComputedBlock
from vllm.utils import Device, get_dtype_size
from vllm.sequence import Sequence, SequenceGroup
from vllm.core.block_v3.registry import BLOCK_MANAGER_REGISTRY
from vllm.core.block_v3.custom_block import AppAwareManager
from vllm.logger import init_logger
from vllm.core.block.block_table import BlockTable
from vllm.core.block_v3.range_utils import intersect_multiple_sets, to_range

logger = init_logger(__name__)

CUSTOM_BLOCK_TABLE = Dict[int, BlockTable]


def require_kv_config_init(func):

    def require_kv_config_init_wrapper(self, *args, **kwargs):
        assert self._initialized, "KV cache config is not initialized"
        return func(self, *args, **kwargs)

    return require_kv_config_init_wrapper


def require_kv_config_not_init(func):

    def require_kv_config_not_init_wrapper(self, *args, **kwargs):
        assert not self._initialized, "KV cache config is already initialized"
        return func(self, *args, **kwargs)

    return require_kv_config_not_init_wrapper


# for simplicity, we assume all components use the same dtype, so that the
# indexing can be easier. can be extended to different dtypes if needed.
def assert_and_get_same_dtype(managers: List[AppAwareManager]):
    dtype = None
    for manager in managers:
        if dtype is None:
            dtype = manager.dtype
        else:
            assert dtype == manager.dtype, "Different dtype detected"
    return dtype


GroupResult = Dict[int, Dict[
    str, List[int]]]  # page_size -> {app_property -> [layer_id]}


def get_num_pages(num_elements: int, page_size: int) -> int:
    return (num_elements + page_size - 1) // page_size


class UniqueID:

    def __init__(self):
        self.names = set()

    def get_group_id(self, group_name: str) -> str:
        i = 0
        while f"{group_name}_{i}" in self.names:
            i += 1
        self.names.add(f"{group_name}_{i}")
        return f"{group_name}_{i}"


@dataclass
class KVCacheScheduleConfig:
    prefix_cache_alignment: Dict[str, int]  # group_id -> alignment


class CustomBlockManager:

    def __init__(self, parallel_config: ParallelConfig,
                 cache_config: CacheConfig):
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self._initialized = False
        # modifying _app_aware_managers is not allowed after is_finalized=True
        self._app_aware_managers: Dict[str, AppAwareManager] = {}
        # reading _kv_cache_config is not allowed before is_finalized=True
        self._kv_cache_config: Optional[KVCacheConfig] = None
        self._kv_cache_schedule_config: Optional[KVCacheScheduleConfig] = None

    @require_kv_config_not_init
    def compile(self, available_cpu_memory: int,
                available_gpu_memory: int) -> KVCacheConfig:
        self._kv_cache_config, self._kv_cache_schedule_config = self._compile_get_kv_cache_config(
            available_gpu_memory)
        # This function is called twice. One with available_gpu_memory=0 to get
        # the config for profile run, and the other with available_gpu_memory>0
        # to get the final KVCacheConfig.
        if available_gpu_memory > 0:
            self._initialized = True
        return self._kv_cache_config

    @property
    @require_kv_config_init
    def kv_cache_config(self):
        assert self._initialized
        return self._kv_cache_config

    @require_kv_config_not_init
    def _compile_get_kv_cache_config(
            self, available_gpu_memory: int) -> KVCacheConfig:
        dtype = assert_and_get_same_dtype(self._app_aware_managers.values())
        available_num_elements = available_gpu_memory // get_dtype_size(dtype)
        group_result: GroupResult = {}
        for layer_id, manager in self._app_aware_managers.items():
            page_size = manager.get_page_size()
            if page_size not in group_result:
                group_result[page_size] = {}
            app_property = manager.get_app_property()
            if app_property not in group_result[page_size]:
                group_result[page_size][app_property] = []
            group_result[page_size][app_property].append(layer_id)

        # only one page size
        if len(group_result) == 1:
            page_size = list(group_result.keys())[0]
            groups = group_result[page_size]  # {app_property -> [layer_id]}
            group_sizes = [len(layers) for layers in groups.values()]
            if self.cache_config.enable_layer_grouping:
                group_size_gcd = math.gcd(*group_sizes)
            else:
                group_size_gcd = 1
            allocator_page_size = page_size * group_size_gcd
            num_pages = get_num_pages(available_num_elements,
                                      allocator_page_size)

            unique_id = UniqueID()

            kv_cache_config = KVCacheConfig(
                buffer_size=num_pages * allocator_page_size,
                buffer_dtype=dtype,
                level0_page_size=allocator_page_size,
                components={},
                block_table_sharing={})
            kv_cache_schedule_config = KVCacheScheduleConfig(
                prefix_cache_alignment={})
            num_pages -= 1  # to avoid overflow due to start_bias

            for app_property, layers in groups.items():
                for i in range(0, len(layers), group_size_gcd):
                    group_id = unique_id.get_group_id(app_property)
                    kv_cache_config.block_table_sharing[group_id] = []
                    for idx_in_group, layer_id in enumerate(
                            layers[i:i + group_size_gcd]):
                        kv_cache_config.block_table_sharing[group_id].append(
                            layer_id)
                        kv_cache_config.components[layer_id] = KVPageType(
                            start_bias=idx_in_group * page_size,
                            num_elements=num_pages * allocator_page_size,
                            page_size=page_size,
                        )
                    layer_id = kv_cache_config.block_table_sharing[group_id][0]
                    kv_cache_schedule_config.prefix_cache_alignment[group_id] = \
                        self._app_aware_managers[layer_id].get_prefix_cache_alignment()
            print("kv_cache_config", kv_cache_config)
            print("kv_cache_schedule_config", kv_cache_schedule_config)
            return kv_cache_config, kv_cache_schedule_config

        raise NotImplementedError("too complex")
        # # all page size have only one app_property, and the app_property is the same:
        # # e.g., spec decode
        # group_result_first = list(group_result.values())[0]
        # if all(len(app_property) == 1 for app_property in group_result.values()) and \
        #     all(app_property.keys() == group_result_first.keys() for app_property in group_result.values()):

    @require_kv_config_not_init
    def add_app_aware_managers(self, managers: Dict[int, AppAwareManager]):
        assert managers.keys() & self._app_aware_managers.keys() == set()
        self._app_aware_managers.update(managers)

    @require_kv_config_not_init
    def add_block_managers_of_model(self,
                                    model: ModelConfig,
                                    parallel_config: ParallelConfig,
                                    prefix: str = ""):
        managers = BLOCK_MANAGER_REGISTRY.get_managers_of_model(
            model, self.cache_config, parallel_config)
        self.add_app_aware_managers({
            f"{prefix}{layer_id}": manager
            for layer_id, manager in managers.items()
        })

    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        total_blocks = 0
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            num_blocks = manager.get_num_required_blocks(
                seq_group, num_lookahead_slots)
            total_blocks += num_blocks
        return total_blocks

    @require_kv_config_init
    def allocate_sequence(
            self, seq_group: SequenceGroup,
            allocator: DeviceAwareBlockAllocator) -> CUSTOM_BLOCK_TABLE:
        block_table: CUSTOM_BLOCK_TABLE = {}

        with (allocator._allocators[Device.GPU].evictor.remove_by_mark_ctx()
              if self.cache_config.enable_prefix_caching else nullcontext()):
            for group_id in self.kv_cache_config.block_table_sharing.keys():
                layer_ids = self.kv_cache_config.block_table_sharing[group_id]
                manager = self._app_aware_managers[layer_ids[0]]
                block = manager.allocate_sequence(seq_group, allocator,
                                                  group_id)
                if block is not None:
                    block.set_block_id_multiplier(len(layer_ids))
                    block_table[group_id] = block
        return block_table

    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(
            self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
            num_lookahead_slots: int) -> int:
        total_blocks = 0
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table
            num_blocks = manager.get_num_blocks_touched_by_append_slots(
                seq, block_table[group_id], num_lookahead_slots)
            total_blocks += num_blocks
        return total_blocks

    @require_kv_config_init
    def append_token_ids(
            self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
            num_lookahead_slots: int,
            last_access_blocks_tracker: LastAccessBlocksTracker) -> int:
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table
            manager.append_token_ids(seq, block_table[group_id],
                                     num_lookahead_slots,
                                     last_access_blocks_tracker)

    @require_kv_config_init
    def get_common_computed_block_ids(
            self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
            computed_blocks_tracker: ComputedBlocksTracker,
            block_allocator: DeviceAwareBlockAllocator) -> ComputedBlock:

        cached_computed_block = computed_blocks_tracker.get_cached_computed_block(
            seq.seq_id)
        if cached_computed_block is not None:
            return cached_computed_block

        # group_id -> [(left, right)], both inclusive
        possible_hit_lens: Dict[str, List[Tuple[int, int]]] = {}

        for group_id in self.kv_cache_config.block_table_sharing.keys():
            block_is_computed =  computed_blocks_tracker.\
                get_cached_block_is_computed(
                seq.seq_id, group_id, block_table[group_id].physical_block_ids)
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table

            possible_hit_lens[group_id] = manager.get_possible_hit_lens(
                block_is_computed)
            # print("possible_hit_lens",
            #       group_id,
            #       to_range(block_is_computed),
            #       possible_hit_lens[group_id],
            #       flush=True)

        intersect_hit_lens = intersect_multiple_sets(
            possible_hit_lens.values())

        hit_len = -1
        for left, right in intersect_hit_lens[::-1]:
            for j in range(right, left - 1, -1):
                valid = all(j % alignment == 0
                            for alignment in self._kv_cache_schedule_config.
                            prefix_cache_alignment.values())
                if valid:
                    hit_len = j
                    break
            if hit_len != -1:
                break
        # print("hit_len", hit_len, flush=True)

        if hit_len == -1:
            hit_len = 0

        computed_blocks: Dict[str, List[int]] = {}

        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            computed_blocks[
                group_id] = manager.filter_computed_blocks_by_token(
                    hit_len, block_table[group_id], block_allocator)

        computed_block = ComputedBlock(computed_blocks, hit_len)
        computed_blocks_tracker.set_cached_compute_block(
            seq.seq_id, computed_block)
        return computed_block

    @require_kv_config_init
    def update_seq_blocks_last_access(
            self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
            last_access_blocks_tracker: LastAccessBlocksTracker):
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table
            manager.update_seq_blocks_last_access(seq, block_table[group_id],
                                                  last_access_blocks_tracker)
