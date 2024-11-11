from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
import math
from typing import List, Optional, Dict, NewType, Tuple, Union

from vllm.config import CacheConfig, ComponentType, KVCacheConfig, KVCacheScheduleConfig, KVCacheScheduleGroupConfig, KVPageType, ModelConfig, ParallelConfig, SchedulerConfig
from vllm.core.block.common import BlockList
from vllm.core.block.cpu_gpu_block_allocator import CpuGpuBlockAllocator
from vllm.core.block.interfaces import (
    Block,
    DeviceAwareBlockAllocator,
)
from vllm.core.block.naive_block import NaiveBlockAllocator, NaiveBlock
from vllm.core.block.prefix_caching_block import ComputedBlocksTracker, LastAccessBlocksTracker, PrefixCachingBlockAllocator
from vllm.core.block_v3.large_block_id_allocator import LargeBlockIDAllocator
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
class LastNumRequiredBlocksInfo:
    seq_id: int
    num_large_blocks_min: Dict[str, int]  # group_id -> num_large_blocks_min
    num_large_blocks_max: Dict[str, int]  # group_id -> num_large_blocks_max


class CustomBlockManager:

    def __init__(self, parallel_config: ParallelConfig,
                 cache_config: CacheConfig, schedule_config: SchedulerConfig):
        self.parallel_config = parallel_config
        self.cache_config = cache_config
        self.scheduler_config = schedule_config
        self._initialized = False
        # modifying _app_aware_managers is not allowed after is_finalized=True
        self._app_aware_managers: Dict[str, AppAwareManager] = {}
        # reading _kv_cache_config is not allowed before is_finalized=True
        self._kv_cache_config: Optional[KVCacheConfig] = None
        self._kv_cache_schedule_config: Optional[KVCacheScheduleConfig] = None
        self.last_num_required_blocks_info = LastNumRequiredBlocksInfo(
            seq_id=-1, num_large_blocks_min={}, num_large_blocks_max={})
        self.global_block_allocator: Union[CpuGpuBlockAllocator,
                                           LargeBlockIDAllocator] = None

    def init_allocator(self):
        kv_cache_config = self._kv_cache_config
        kv_cache_schedule_config = self._kv_cache_schedule_config
        self.num_total_gpu_blocks = kv_cache_schedule_config.num_level0_pages
        self.num_total_cpu_blocks = 0

        for group_id, layer_ids in kv_cache_config.block_table_sharing.items():
            for layer_id in layer_ids:
                self._app_aware_managers[layer_id].attach_group_id(group_id)

        if self.scheduler_config.enable_two_level_page:
            self.global_block_allocator = LargeBlockIDAllocator(
                num_blocks=self.num_total_gpu_blocks)
            if self.cache_config.enable_prefix_caching:
                self.group_allocators = {}
                for group_id, layer_ids in kv_cache_config.block_table_sharing.items(
                ):
                    schedule_component = kv_cache_schedule_config.groups[
                        group_id]
                    allocator = PrefixCachingBlockAllocator(
                        num_blocks=self.num_total_gpu_blocks *
                        schedule_component.large_small_ratio,
                        block_size=schedule_component.block_size,
                        owned_block_ids=[],
                        enable_two_level_page=True,
                        two_level_kwargs={
                            'large_small_ratio':
                            schedule_component.large_small_ratio,
                            'large_block_id_allocator':
                            self.global_block_allocator,
                        },
                        allocator_type="prefix_caching",
                    )
                    # TODO: only allocate null block when needed
                    allocator._hashless_allocator._block_id_allocator.set_new_large_block_quota(
                        1)
                    allocator.allocate_or_get_null_block()
                    allocator._hashless_allocator._block_id_allocator.set_new_large_block_quota(
                        0)
                    self.group_allocators[group_id] = allocator
            else:
                self.group_allocators = {}

                for group_id, layer_ids in kv_cache_config.block_table_sharing.items(
                ):
                    schedule_component = kv_cache_schedule_config.groups[
                        group_id]
                    allocator = NaiveBlockAllocator(
                        create_block=NaiveBlock,  # type: ignore
                        num_blocks=self.num_total_gpu_blocks *
                        schedule_component.large_small_ratio,
                        block_size=schedule_component.block_size,
                        block_ids=[],
                        enable_two_level_page=True,
                        two_level_kwargs={
                            'large_small_ratio':
                            schedule_component.large_small_ratio,
                            'large_block_id_allocator':
                            self.global_block_allocator,
                        },
                        allocator_type="naive")
                    # TODO: only allocate null block when needed
                    allocator._block_id_allocator.set_new_large_block_quota(1)
                    allocator.allocate_or_get_null_block()
                    allocator._block_id_allocator.set_new_large_block_quota(0)
                    self.group_allocators[group_id] = allocator
        else:
            self.global_block_allocator = CpuGpuBlockAllocator.create(
                allocator_type="prefix_caching"
                if self.cache_config.enable_prefix_caching else "naive",
                num_gpu_blocks=self.num_total_gpu_blocks,
                num_cpu_blocks=self.num_total_cpu_blocks,
                block_size=self.cache_config.block_size,
            )
            self.group_allocators = {}

        if self.scheduler_config.enable_two_level_page:
            self._computed_blocks_tracker = ComputedBlocksTracker(
                self.group_allocators, enable_two_level_page=True)
            self._last_access_blocks_tracker = LastAccessBlocksTracker(
                self.group_allocators, enable_two_level_page=True)
        else:
            self._computed_blocks_tracker = ComputedBlocksTracker(
                self.global_block_allocator, enable_two_level_page=False
            )  # same for all groups, dispatch group_ids inside
            self._last_access_blocks_tracker = LastAccessBlocksTracker(
                self.global_block_allocator,
                enable_two_level_page=False)  # same for all groups

        logger.info(
            "############### create PerlayerBlockSpaceManager, block_size: {}, page size: {}, num pages: {} ###############"
            .format(
                self.cache_config.block_size,
                self.kv_cache_config.level0_page_size,
                self.num_total_gpu_blocks,
            ))

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
            self.init_allocator()
        return self._kv_cache_config

    @property
    @require_kv_config_init
    def kv_cache_config(self):
        assert self._initialized
        return self._kv_cache_config

    def _compile_two_level(
            self, group_result: GroupResult, available_num_elements: int,
            dtype: ComponentType
    ) -> Tuple[KVCacheConfig, KVCacheScheduleConfig]:
        group_layer_mapping = {}  # {app_property -> [layer_id]}}
        group_page_size = {}  # {app_property -> page_size_of_group}
        layer_page_size = {}  # {layer_id -> page_size_of_layer}
        for page_size, group_ids in group_result.items():
            for app_property, layer_ids in group_ids.items():
                assert app_property not in group_layer_mapping
                group_layer_mapping[app_property] = layer_ids
                group_page_size[app_property] = page_size * len(layer_ids)
                for layer_id in layer_ids:
                    layer_page_size[layer_id] = page_size

        level0_page_size = math.lcm(*group_page_size.values())
        num_pages = get_num_pages(available_num_elements, level0_page_size)

        kv_cache_config = KVCacheConfig(buffer_size=num_pages *
                                        level0_page_size,
                                        buffer_dtype=dtype,
                                        level0_page_size=level0_page_size,
                                        components={},
                                        block_table_sharing={})
        num_pages -= 1  # to avoid overflow due to start_bias
        kv_cache_schedule_config = KVCacheScheduleConfig(
            groups={}, num_level0_pages=num_pages)

        for app_property, layer_ids in group_layer_mapping.items():
            group_id = app_property
            kv_cache_config.block_table_sharing[group_id] = layer_ids
            for idx_in_group, layer_id in enumerate(layer_ids):
                kv_cache_config.components[layer_id] = KVPageType(
                    start_bias=idx_in_group * layer_page_size[layer_id],
                    num_elements=num_pages * level0_page_size,
                    page_size=layer_page_size[layer_id],
                )
                if idx_in_group == 0:
                    kv_cache_schedule_config.groups[
                        group_id] = KVCacheScheduleGroupConfig(
                            prefix_cache_alignment=self._app_aware_managers[
                                layer_id].get_prefix_cache_alignment(),
                            block_size=self._app_aware_managers[layer_id].
                            get_block_size(),
                            large_small_ratio=level0_page_size //
                            group_page_size[app_property],
                        )

        print("num_pages", num_pages)
        print("kv_cache_config", kv_cache_config)
        print("kv_cache_schedule_config", kv_cache_schedule_config)
        return kv_cache_config, kv_cache_schedule_config

    def _compile_single_page_size(
            self, group_result: GroupResult, available_num_elements: int,
            dtype: ComponentType
    ) -> Tuple[KVCacheConfig, KVCacheScheduleConfig]:
        page_size = list(group_result.keys())[0]
        groups = group_result[page_size]  # {app_property -> [layer_id]}
        group_sizes = [len(layers) for layers in groups.values()]
        if self.cache_config.enable_layer_grouping:
            group_size_gcd = math.gcd(*group_sizes)
        else:
            group_size_gcd = 1
        allocator_page_size = page_size * group_size_gcd
        num_pages = get_num_pages(available_num_elements, allocator_page_size)

        unique_id = UniqueID()

        kv_cache_config = KVCacheConfig(buffer_size=num_pages *
                                        allocator_page_size,
                                        buffer_dtype=dtype,
                                        level0_page_size=allocator_page_size,
                                        components={},
                                        block_table_sharing={})
        num_pages -= 1  # to avoid overflow due to start_bias
        kv_cache_schedule_config = KVCacheScheduleConfig(
            groups={}, num_level0_pages=num_pages)

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
                if idx_in_group == 0:
                    kv_cache_schedule_config.groups[
                        group_id] = KVCacheScheduleGroupConfig(
                            prefix_cache_alignment=self._app_aware_managers[
                                layer_id].get_prefix_cache_alignment(),
                            block_size=self._app_aware_managers[layer_id].
                            get_block_size(),
                        )
        print("num_pages", num_pages)
        print("kv_cache_config", kv_cache_config)
        print("kv_cache_schedule_config", kv_cache_schedule_config)
        return kv_cache_config, kv_cache_schedule_config

    @require_kv_config_not_init
    def _compile_get_kv_cache_config(
        self, available_gpu_memory: int
    ) -> Tuple[KVCacheConfig, KVCacheScheduleConfig]:
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

        assert self.scheduler_config.use_per_layer_block_manager
        if self.scheduler_config.enable_two_level_page:
            return self._compile_two_level(group_result,
                                           available_num_elements, dtype)
        else:
            assert len(group_result) == 1
            return self._compile_single_page_size(group_result,
                                                  available_num_elements,
                                                  dtype)

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
    def get_num_required_block_small_to_large(
            self, num_required_blocks: Dict[str, int],
            seq_id: int) -> Tuple[Dict[str, int], Dict[str, int]]:
        num_large_blocks_min: Dict[str, int] = {}
        num_large_blocks_max: Dict[str, int] = {}
        for group_id, num_required_small_blocks in num_required_blocks.items():
            schedule_config = self._kv_cache_schedule_config.groups[group_id]
            # TODO: more precise calculation for append_slots
            num_large_blocks_max[group_id] = math.ceil(
                num_required_small_blocks / schedule_config.large_small_ratio)

            num_new_small_blocks_min = num_required_small_blocks - self.group_allocators[
                group_id].get_num_free_small_blocks()
            num_new_small_blocks_min = max(num_new_small_blocks_min, 0)
            num_large_blocks_min[group_id] = math.ceil(
                num_new_small_blocks_min / schedule_config.large_small_ratio)

        self.last_num_required_blocks_info = LastNumRequiredBlocksInfo(
            seq_id=seq_id,
            num_large_blocks_min=num_large_blocks_min,
            num_large_blocks_max=num_large_blocks_max,
        )
        return num_large_blocks_min, num_large_blocks_max

    def set_large_block_quota(self, seq_id: int):
        assert self.last_num_required_blocks_info.seq_id == seq_id
        block_info = self.last_num_required_blocks_info
        total_large_blocks_min = sum(block_info.num_large_blocks_min.values())
        total_large_blocks_max = sum(block_info.num_large_blocks_max.values())
        num_free_large_blocks = self.global_block_allocator.get_num_free_blocks(
        )
        if num_free_large_blocks >= total_large_blocks_max:
            new_large_block_quota = block_info.num_large_blocks_max
        elif num_free_large_blocks >= total_large_blocks_min:
            new_large_block_quota = block_info.num_large_blocks_min
        else:
            raise ValueError("Not enough free large blocks for the sequence")

        for group_id in self.kv_cache_config.block_table_sharing.keys():
            allocator = self.group_allocators[group_id]
            allocator.set_new_large_block_quota(
                new_large_block_quota[group_id])

    @require_kv_config_init
    def get_num_required_blocks(self,
                                seq_group: SequenceGroup,
                                num_lookahead_slots: int = 0) -> int:
        num_required_blocks: Dict[str,
                                  int] = {}  # group_id -> num_required_blocks
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            num_blocks = manager.get_num_required_blocks(
                seq_group, num_lookahead_slots)
            num_required_blocks[group_id] = num_blocks

        if self.scheduler_config.enable_two_level_page:
            num_large_blocks_min, num_large_blocks_max = self.get_num_required_block_small_to_large(
                num_required_blocks, seq_group.seqs[0].seq_id)
            total_blocks = sum(num_large_blocks_min.values())
        else:
            total_blocks = sum(num_required_blocks.values())
        return total_blocks

    @require_kv_config_init
    def allocate_sequence(self,
                          seq_group: SequenceGroup) -> CUSTOM_BLOCK_TABLE:
        block_table: CUSTOM_BLOCK_TABLE = {}
        seq_id = seq_group.seqs[0].seq_id

        if self.scheduler_config.enable_two_level_page:
            self.set_large_block_quota(seq_id)
            for group_allocator in self.group_allocators.values():
                group_allocator.add_req_for_two_level_page(seq_id)

        with ExitStack() as ctx_stack:
            if self.cache_config.enable_prefix_caching:
                if self.scheduler_config.enable_two_level_page:
                    for allocator in self.group_allocators.values():
                        ctx_stack.enter_context(
                            allocator.evictor.remove_by_mark_ctx())
                else:
                    ctx_stack.enter_context(
                        self.global_block_allocator._allocators[
                            Device.GPU].evictor.remove_by_mark_ctx())
            for group_id in self.kv_cache_config.block_table_sharing.keys():
                layer_ids = self.kv_cache_config.block_table_sharing[group_id]
                manager = self._app_aware_managers[layer_ids[0]]
                if self.scheduler_config.enable_two_level_page:
                    allocator = self.group_allocators[group_id]
                else:
                    allocator = self.global_block_allocator
                block = manager.allocate_sequence(seq_group, allocator,
                                                  group_id)
                if block is not None:
                    block.set_block_id_multiplier(len(layer_ids))
                    block_table[group_id] = block

        if self.cache_config.enable_prefix_caching:
            self._computed_blocks_tracker.add_seq(seq_id)
            self._last_access_blocks_tracker.add_seq(seq_id)
        return block_table

    @require_kv_config_init
    def get_num_blocks_touched_by_append_slots(
            self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
            num_lookahead_slots: int) -> int:
        num_required_blocks: Dict[str,
                                  int] = {}  # group_id -> num_required_blocks
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table
            num_blocks = manager.get_num_blocks_touched_by_append_slots(
                seq, block_table[group_id], num_lookahead_slots)
            num_required_blocks[group_id] = num_blocks

        if self.scheduler_config.enable_two_level_page:
            num_large_blocks_min, num_large_blocks_max = self.get_num_required_block_small_to_large(
                num_required_blocks, seq.seq_id)
            total_blocks = sum(num_large_blocks_min.values())
        else:
            total_blocks = sum(num_required_blocks.values())
        return total_blocks

    @require_kv_config_init
    def append_token_ids(self, seq: Sequence, block_table: CUSTOM_BLOCK_TABLE,
                         num_lookahead_slots: int) -> int:
        if self.scheduler_config.enable_two_level_page:
            self.set_large_block_quota(seq.seq_id)
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table
            manager.append_token_ids(seq, block_table[group_id],
                                     num_lookahead_slots)

    @require_kv_config_init
    def get_common_computed_block_ids(
            self, seq: Sequence,
            block_table: CUSTOM_BLOCK_TABLE) -> ComputedBlock:

        cached_computed_block = self._computed_blocks_tracker.get_cached_computed_block(
            seq.seq_id)
        if cached_computed_block is not None:
            return cached_computed_block

        # group_id -> [(left, right)], both inclusive
        possible_hit_lens: Dict[str, List[Tuple[int, int]]] = {}

        for group_id in self.kv_cache_config.block_table_sharing.keys():
            block_is_computed =  self._computed_blocks_tracker.\
                get_cached_block_is_computed(
                seq.seq_id, group_id, block_table[group_id].physical_block_ids)
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table

            possible_hit_lens[group_id] = manager.get_possible_hit_lens(
                block_is_computed)
            print("possible_hit_lens",
                  group_id,
                  to_range(block_is_computed),
                  possible_hit_lens[group_id],
                  flush=True)

        intersect_hit_lens = intersect_multiple_sets(
            possible_hit_lens.values())

        hit_len = -1
        for left, right in intersect_hit_lens[::-1]:
            for j in range(right, left - 1, -1):
                valid = all(
                    j % g.prefix_cache_alignment == 0
                    for g in self._kv_cache_schedule_config.groups.values())
                if valid:
                    hit_len = j
                    break
            if hit_len != -1:
                break
        print("hit_len", hit_len, flush=True)

        if hit_len == -1:
            hit_len = 0

        computed_blocks: Dict[str, List[int]] = {}

        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            if self.scheduler_config.enable_two_level_page:
                allocator = self.group_allocators[group_id]
            else:
                allocator = self.global_block_allocator
            computed_blocks[
                group_id] = manager.filter_computed_blocks_by_token(
                    hit_len, block_table[group_id], allocator)

        computed_block = ComputedBlock(computed_blocks, hit_len)
        self._computed_blocks_tracker.set_cached_compute_block(
            seq.seq_id, computed_block)
        return computed_block

    @require_kv_config_init
    def update_seq_blocks_last_access(self, seq: Sequence,
                                      block_table: CUSTOM_BLOCK_TABLE):
        if self.cache_config.enable_prefix_caching:
            for group_id in self.kv_cache_config.block_table_sharing.keys():
                manager = self._app_aware_managers[
                    self.kv_cache_config.block_table_sharing[group_id][0]]
                assert group_id in block_table
                manager.update_seq_blocks_last_access(
                    seq, block_table[group_id],
                    self._last_access_blocks_tracker)

    @require_kv_config_init
    def free_skipped_blocks(self, seq: Sequence,
                            block_table: CUSTOM_BLOCK_TABLE):
        for group_id in self.kv_cache_config.block_table_sharing.keys():
            manager = self._app_aware_managers[
                self.kv_cache_config.block_table_sharing[group_id][0]]
            assert group_id in block_table
            manager.free_skipped_blocks(seq, block_table[group_id],
                                        self._last_access_blocks_tracker)

    @require_kv_config_init
    def trackers_free(self, seq_id: int):
        if self.cache_config.enable_prefix_caching:
            self._last_access_blocks_tracker.remove_seq(seq_id)
            self._computed_blocks_tracker.remove_seq(seq_id)
        for group_allocator in self.group_allocators.values():
            group_allocator.remove_seq(seq_id)

    @require_kv_config_init
    def access_all_blocks_in_seq(self, seq: Sequence, now: float):
        if self.cache_config.enable_prefix_caching:
            # Record the latest access time for the sequence. The actual update
            # of the block ids is deferred to the sequence free(..) call, since
            # only during freeing of block ids, the blocks are actually added to
            # the evictor (which is when the most updated time is required)
            # (This avoids expensive calls to mark_blocks_as_accessed(..))
            self._last_access_blocks_tracker.update_last_access(
                seq.seq_id, now)

    @require_kv_config_init
    def mark_blocks_as_computed(self):
        if self.cache_config.enable_prefix_caching:
            if self.scheduler_config.enable_two_level_page:
                for allocator in self.group_allocators.values():
                    allocator.mark_blocks_as_computed([])
            else:
                self.global_block_allocator.mark_blocks_as_computed([])

    @require_kv_config_init
    def get_prefix_cache_hit_rate(self, device: Device) -> float:
        return self.global_block_allocator.get_prefix_cache_hit_rate(device)

    @require_kv_config_init
    def confirm_all_remove(self):
        if self.cache_config.enable_prefix_caching:
            if self.scheduler_config.enable_two_level_page:
                for allocator in self.group_allocators.values():
                    allocator.evictor.confirm_all_remove()
            else:
                self.global_block_allocator._allocators[
                    Device.GPU].evictor.confirm_all_remove()
