from contextlib import contextmanager
from dataclasses import dataclass
import enum
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Dict, List, OrderedDict, Tuple
from sortedcontainers import SortedDict, SortedSet
import math
if TYPE_CHECKING:
    from vllm.core.block.prefix_caching_block import PrefixCachingBlockAllocator


class EvictionPolicy(enum.Enum):
    """Enum for eviction policy used by make_evictor to instantiate the correct
       Evictor subclass.
    """
    LRU = enum.auto()


class Evictor(ABC):
    """The Evictor subclasses should be used by the BlockAllocator class to
    handle eviction of freed PhysicalTokenBlocks.
    """

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __contains__(self, block_id: int) -> bool:
        pass

    @abstractmethod
    def evict(self) -> Tuple[int, int]:
        """Runs the eviction algorithm and returns the evicted block's
        content hash along with physical block id along with physical block id
        """
        pass

    @abstractmethod
    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        """Adds block to the evictor, making it a candidate for eviction"""
        pass

    @abstractmethod
    def update(self, block_id: int, last_accessed: float):
        """Update corresponding block's access time in metadata"""
        pass

    @abstractmethod
    def remove(self, block_id: int):
        """Remove a given block id from the cache."""
        pass

    @property
    @abstractmethod
    def num_blocks(self) -> int:
        pass


@dataclass
class BlockMetaData():
    """Data structure for storing key data describe cached block, so that
    evitor could use to make its decision which one to choose for eviction

    Here we use physical block id as the dict key, as there maybe several
    blocks with the same content hash, but their physical id is unique.
    """

    def __init__(self, content_hash: int, num_hashed_tokens: int,
                 last_accessed: float, block_id: int):
        self.content_hash = content_hash
        self.num_hashed_tokens = num_hashed_tokens
        self.last_accessed = last_accessed
        self.block_id = block_id


class LRUEvictor(Evictor):
    """Evicts in a least-recently-used order using the last_accessed timestamp
    that's recorded in the PhysicalTokenBlock. If there are multiple blocks with
    the same last_accessed time, then the one with the largest num_hashed_tokens
    will be evicted. If two blocks each have the lowest last_accessed time and
    highest num_hashed_tokens value, then one will be chose arbitrarily
    """

    def __init__(self):
        self.free_table: OrderedDict[
            int, BlockMetaData] = OrderedDict()  # hash -> block_meta
        self.sorted_table: SortedSet[Tuple[int, int, int]] = SortedSet(
        )  # (last_accessed, -num_hashed_tokens, block_id) -> hash
        self._num_blocks = 0
        self.mark_removed: set[int] = set()
        self.remove_by_mark = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    def evict(self) -> Tuple[int, int]:
        if self._num_blocks <= 0:
            raise ValueError("No usable cache memory left")

        evicted_block_id = self.sorted_table.pop(0)[2]
        evicted_block = self.free_table.pop(evicted_block_id)
        self._num_blocks -= 1

        return evicted_block.block_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self._num_blocks += 1
        if last_accessed == -1:
            # if block_id == 3697:
            #     print("==============evictor recover block_id:", id(self),
            #           block_id, content_hash, num_hashed_tokens, last_accessed)
            #     import traceback
            #     traceback.print_stack()
            # a hack: sliding window may alloc a block and then free it without using it
            # not remove it from the free_table, so that keep the orignal position
            assert block_id in self.free_table, f"block_id: {block_id} not in free_table"
            assert block_id in self.mark_removed, f"block_id: {block_id} not in mark_removed"
            block_meta = self.free_table[block_id]
            assert block_meta.content_hash == content_hash, f"block_id: {block_id} content_hash not match"
            assert block_meta.num_hashed_tokens == num_hashed_tokens
            self.mark_removed.remove(block_id)
            self.sorted_table.add(
                (block_meta.last_accessed, -num_hashed_tokens, block_id))
            return
        # if block_id == 3697:
        #     print("==============evictor add block_id:", block_id,
        #           content_hash, num_hashed_tokens, last_accessed)
        #     import traceback
        #     traceback.print_stack()
        assert block_id not in self.mark_removed
        if block_id in self.free_table:
            # to upate the insert time in ordered dict
            block_meta = self.free_table.pop(block_id)
            key = (block_meta.last_accessed, -block_meta.num_hashed_tokens,
                   block_meta.block_id)
            self.sorted_table.discard(key)
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, block_id)
        self.sorted_table.add((last_accessed, -num_hashed_tokens, block_id))

    def update(self, block_id: int, last_accessed: float):
        raise AssertionError("This line should be unreachable.")
        assert self.free_table[block_id].alive
        self.free_table[block_id].last_accessed = last_accessed
        print("==============evictor update block_id:", block_id,
              last_accessed)

    def remove(self, block_id: int):
        # if block_id == 3697:
        #     print("==============evictor remove block_id:", id(self), block_id,
        #           self.remove_by_mark)
        #     import traceback
        #     traceback.print_stack()
        self._num_blocks -= 1
        if block_id not in self.free_table:
            raise ValueError(
                f"Attempting to remove block {block_id} that's not in the evictor"
            )
        block_meta = self.free_table[block_id]

        key = (block_meta.last_accessed, -block_meta.num_hashed_tokens,
               block_meta.block_id)
        self.sorted_table.discard(key)

        if self.remove_by_mark:
            self.mark_removed.add(block_id)
        else:
            self.free_table.pop(block_id)

        return block_id, block_meta.content_hash

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def confirm_all_remove(self):
        for _id in self.mark_removed:
            self.free_table.pop(_id)
            # if _id == 3697:
            #     print("==============evictor confirm_all_remove block_id:",
            #           id(self), _id)
            #     import traceback
            #     traceback.print_stack()
        self.mark_removed.clear()

    @contextmanager
    def remove_by_mark_ctx(self):
        prev_remove_by_mark = self.remove_by_mark
        self.remove_by_mark = True
        try:
            yield
        finally:
            self.remove_by_mark = prev_remove_by_mark

    def level0_lru(self) -> Tuple[int, int]:
        raise NotImplementedError

    def evict_level0(self):
        raise NotImplementedError


class Level1LRUEvictor(LRUEvictor):

    def __init__(self, large_small_ratio: int, **kwargs):
        super().__init__()
        self.large_small_ratio = large_small_ratio
        # large_block_id -> num_free_small_blocks
        self.free_block_counter: Dict[int, int] = {}
        self.free_level0_blocks: SortedSet[Tuple[int, int, int]] = SortedSet()
        self.free_level0_table: Dict[int, BlockMetaData] = {
        }  # block_id -> block_meta

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        super().add(block_id, content_hash, num_hashed_tokens, last_accessed)
        lv0_block_id = block_id // self.large_small_ratio
        if lv0_block_id not in self.free_block_counter:
            self.free_block_counter[lv0_block_id] = 0
        self.free_block_counter[lv0_block_id] += 1
        if self.free_block_counter[lv0_block_id] == self.large_small_ratio:
            block_ids = [
                lv0_block_id * self.large_small_ratio + i
                for i in range(self.large_small_ratio)
            ]
            lru = (-math.inf, -math.inf)  # last_accessed, -num_hashed_tokens
            for block_id in block_ids:
                block_meta = self.free_table[block_id]
                lru = max(
                    lru,
                    (block_meta.last_accessed, -block_meta.num_hashed_tokens))
            # if lv0_block_id == 84:
            #     print("add in Level1LRUEvictor:",
            #           (lru[0], lru[1], lv0_block_id))
            self.free_level0_blocks.add((lru[0], lru[1], lv0_block_id))
            self.free_level0_table[lv0_block_id] = BlockMetaData(
                content_hash, -lru[1], lru[0], lv0_block_id)

    def evict_block_id(self, block_id: int):
        raise ValueError("should not be called")

    def evict(self):
        raise ValueError("should not be called. Use evict_level0 instead")

    def update(self, block_id: int, last_accessed: float):
        raise ValueError("should not be called")

    def remove_level1_block_id(self, lv1_block_id: int):
        lv0_block_id = lv1_block_id // self.large_small_ratio
        # if lv1_block_id == 84:
        #     print(
        #         "remove_level1_block_id:", lv1_block_id, lv0_block_id,
        #         self.free_block_counter[lv0_block_id],
        #         self.free_block_counter[lv0_block_id] ==
        #         self.large_small_ratio)
        if self.free_block_counter[lv0_block_id] == self.large_small_ratio:
            block_meta = self.free_level0_table.pop(lv0_block_id)
            key = (block_meta.last_accessed, -block_meta.num_hashed_tokens,
                   lv0_block_id)
            # if lv0_block_id == 84:
            #     print("to remove", key)
            assert key in self.free_level0_blocks, "key {} not in free_level0_blocks".format(
                key)
            self.free_level0_blocks.discard(key)
        self.free_block_counter[lv0_block_id] -= 1

    def remove(self, lv1_block_id):
        super().remove(lv1_block_id)
        self.remove_level1_block_id(lv1_block_id)
        # TODO: handle level0 budget

    @property
    def num_blocks(self) -> int:
        return super(
        ).num_blocks - self.num_lv0_blocks * self.large_small_ratio

    def level0_lru(self) -> Tuple[int, int]:
        block = next(iter(self.free_level0_blocks))
        return block[0], block[1]

    @property
    def num_lv0_blocks(self):
        return len(self.free_level0_blocks)

    def evict_level0(
            self) -> Tuple[int, List[int]]:  # lv0_block_id, lv1_block_ids
        if len(self.free_level0_blocks) == 0:
            raise ValueError("No usable cache memory left")
        evicted_block = iter(self.free_level0_blocks).__next__()
        lv0_block_id = evicted_block[2]
        # if lv0_block_id == 84:
        #     print("evict in evict_level0:", lv0_block_id)

        lv1_block_ids = [
            lv0_block_id * self.large_small_ratio + i
            for i in range(self.large_small_ratio)
        ]
        for lv1_block_id in lv1_block_ids:
            self.remove(lv1_block_id)
        return lv0_block_id, lv1_block_ids

    def evict_level1(self) -> Tuple[int, int]:
        if self._num_blocks <= 0:
            raise ValueError("No usable cache memory left")
        lv1_block_id, content_hash = super().evict()
        self.remove_level1_block_id(lv1_block_id)

        return lv1_block_id, content_hash


def make_evictor(eviction_policy: EvictionPolicy, enable_two_level_page: bool,
                 two_level_kwargs) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        if enable_two_level_page:
            return Level1LRUEvictor(**two_level_kwargs)
        else:
            return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")


class Level0LRUEvictor:

    def __init__(self, level1_evictors: Dict[str, Level1LRUEvictor],
                 level1_allocators: Dict[str, "PrefixCachingBlockAllocator"]):
        self.level1_evictors: Dict[str, Level1LRUEvictor] = level1_evictors
        self.level1_allocators: Dict[
            str, "PrefixCachingBlockAllocator"] = level1_allocators
        for allocator in level1_allocators.values():
            assert allocator.level0_evictor is None
            allocator.level0_evictor = self

    def evict_level0(self):
        worst_evictor: str = ""
        worst_lru = (math.inf, 0)  # last_accessed, -num_hashed_tokens
        for evictor_name, evictor in self.level1_evictors.items():
            if evictor.num_lv0_blocks == 0:
                continue
            lru = evictor.level0_lru()
            if lru < worst_lru:
                worst_lru = lru
                worst_evictor = evictor_name

        assert worst_evictor != ""
        lv0_block_id, lv1_block_ids = self.level1_evictors[
            worst_evictor].evict_level0()
        self.level1_allocators[worst_evictor].remove_blocks_from_evictor(
            lv1_block_ids)

    @property
    def num_lv0_blocks(self):
        return sum(evictor.num_lv0_blocks
                   for evictor in self.level1_evictors.values())
