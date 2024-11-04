from contextlib import contextmanager
from dataclasses import dataclass
import enum
from abc import ABC, abstractmethod
from typing import OrderedDict, Tuple
from sortedcontainers import SortedDict


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
        self.alive = True


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
        self.sorted_table: SortedDict[(int, int, int), int] = SortedDict(
        )  # (last_accessed, -num_hashed_tokens, block_id) -> hash
        self._num_blocks = 0
        self.mark_removed: set[int] = set()
        self.remove_by_mark = False

    def __contains__(self, block_id: int) -> bool:
        return block_id in self.free_table

    # force eviction, for testing only
    def evict_block_id(self, block_id: int):
        # print("Evicting block_id:", block_id)
        block = self.free_table.pop(block_id)
        self.sorted_table.pop(
            (block.last_accessed, -block.num_hashed_tokens, block.block_id))
        return block_id, block.content_hash

    def evict(self) -> Tuple[int, int]:
        if self._num_blocks <= 0:
            raise ValueError("No usable cache memory left")

        evicted_block_id = self.sorted_table.popitem(0)[0][2]
        evicted_block = self.free_table.pop(evicted_block_id)
        self._num_blocks -= 1

        return evicted_block.block_id, evicted_block.content_hash

    def add(self, block_id: int, content_hash: int, num_hashed_tokens: int,
            last_accessed: float):
        self._num_blocks += 1
        if last_accessed == -1:
            # if block_id == 3697:
            #     print("==============evictor recover block_id:", block_id,
            #           content_hash, num_hashed_tokens, last_accessed)
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
            self.sorted_table[(block_meta.last_accessed, -num_hashed_tokens,
                               block_id)] = content_hash
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
            self.sorted_table.pop(
                (block_meta.last_accessed, -block_meta.num_hashed_tokens,
                 block_meta.block_id))
        self.free_table[block_id] = BlockMetaData(content_hash,
                                                  num_hashed_tokens,
                                                  last_accessed, block_id)
        self.sorted_table[(last_accessed, -num_hashed_tokens,
                           block_id)] = content_hash

    def update(self, block_id: int, last_accessed: float):
        raise AssertionError("This line should be unreachable.")
        assert self.free_table[block_id].alive
        self.free_table[block_id].last_accessed = last_accessed
        print("==============evictor update block_id:", block_id,
              last_accessed)

    def remove(self, block_id: int):
        # if block_id == 3697:
        #     print("==============evictor remove block_id:", block_id,
        #           self.remove_by_mark)
        #     import traceback
        #     traceback.print_stack()
        self._num_blocks -= 1
        if block_id not in self.free_table:
            raise ValueError(
                "Attempting to remove block that's not in the evictor")
        block_meta = self.free_table[block_id]

        self.sorted_table.pop(
            (block_meta.last_accessed, -block_meta.num_hashed_tokens,
             block_meta.block_id))

        if self.remove_by_mark:
            self.mark_removed.add(block_id)
        else:
            self.free_table.pop(block_id)

    @property
    def num_blocks(self) -> int:
        return self._num_blocks

    def confirm_all_remove(self):
        for _id in self.mark_removed:
            self.free_table.pop(_id)
            # if _id == 3697:
            #     print("==============evictor confirm_all_remove block_id:",
            #           _id)
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


def make_evictor(eviction_policy: EvictionPolicy) -> Evictor:
    if eviction_policy == EvictionPolicy.LRU:
        return LRUEvictor()
    else:
        raise ValueError(f"Unknown cache eviction policy: {eviction_policy}")
