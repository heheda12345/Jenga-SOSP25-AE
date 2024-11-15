from collections import OrderedDict
from typing import Dict, List, Set, Tuple

from vllm.core.block.common import RefCounter
from vllm.core.block.null_block import NULL_BLOCK_SEQ_ID
from vllm.core.block_v3.large_block_id_allocator import LargeBlockIDAllocator


class SmallBlockIDAllocator:

    def __init__(self, ref_counter: RefCounter,
                 large_block_id_allocator: LargeBlockIDAllocator,
                 large_small_ratio: int):
        self.large_small_ratio = large_small_ratio

        # update when alloc/free small blocks

        # small_block_id
        self.free_block_id_no_prefer: Set[int] = set()
        # seq_id -> small_block_id
        self.free_block_id_with_prefer: Dict[int, OrderedDict[int, None]] = {}
        # large_block_id -> num_free_small_blocks
        self.free_block_counter: Dict[int, int] = {}
        self.num_free_small_blocks = 0

        # update when alloc/free large blocks

        self.large_block_id_allocator = large_block_id_allocator
        self.new_large_block_quota = 0
        self.ref_counter = ref_counter
        # large_block_id -> seq_id. Each page has a prefer seq_id, even when the seq is freed.
        self.large_block_to_seq_id: Dict[int, int] = {}
        self.add_seq(NULL_BLOCK_SEQ_ID)

    class NoFreeLevel1BlocksError(ValueError):
        pass

    def set_new_large_block_quota(self, new_large_block_quota: int):
        self.new_large_block_quota = new_large_block_quota

    def allocate_block_id(
            self,
            seq_id: int,
            affine_only: bool = False
    ) -> Tuple[int, List[int]]:  # small_block_id
        if len(self.free_block_id_with_prefer[seq_id]) == 0:
            if self.new_large_block_quota > 0:
                # allocate new large block
                new_large_block_id = self.large_block_id_allocator.allocate_block_id(
                )
                # update large block trackers
                new_small_block_ids = [
                    new_large_block_id * self.large_small_ratio + i
                    for i in range(self.large_small_ratio)
                ]
                self.ref_counter.add_new_blocks(new_small_block_ids)
                self.large_block_to_seq_id[new_large_block_id] = seq_id
                self.new_large_block_quota -= 1
                assert self.new_large_block_quota >= 0
                # update small block trackers
                self.free_block_id_with_prefer[seq_id].update(
                    (i, None) for i in new_small_block_ids)
                self.free_block_counter[
                    new_large_block_id] = self.large_small_ratio
                self.num_free_small_blocks += self.large_small_ratio
                # allocate small block
                small_block_id = self.free_block_id_with_prefer[
                    seq_id].popitem(last=False)[0]

            elif self.free_block_id_no_prefer:
                # use small blocks in no_prefer list
                small_block_id = self.free_block_id_no_prefer.pop()
                lv0_block_id = small_block_id // self.large_small_ratio

                # add all free small blocks in this large block to seq_id's prefer list
                # assert lv0_block_id not in self.large_block_to_seq_id
                all_small_block_ids = set(
                    lv0_block_id * self.large_small_ratio + i
                    for i in range(self.large_small_ratio))
                all_small_block_ids = all_small_block_ids & self.free_block_id_no_prefer
                self.free_block_id_with_prefer[seq_id].update(
                    (i, None) for i in all_small_block_ids)
                self.large_block_to_seq_id[lv0_block_id] = seq_id

            elif not affine_only:
                for victim_seq_id, small_block_ids in self.free_block_id_with_prefer.items(
                ):
                    if small_block_ids:
                        small_block_id = small_block_ids.popitem(last=False)[0]
                        break
                else:
                    raise SmallBlockIDAllocator.NoFreeLevel1BlocksError()
            else:
                raise SmallBlockIDAllocator.NoFreeLevel1BlocksError()
        else:
            small_block_id = self.free_block_id_with_prefer[seq_id].popitem(
                last=False)[0]

        self.free_block_counter[small_block_id // self.large_small_ratio] -= 1
        assert self.free_block_counter[small_block_id //
                                       self.large_small_ratio] >= 0
        self.num_free_small_blocks -= 1
        return small_block_id

    def free_block_id(self, block_id: int):  # small_block_id
        lv0_block_id = block_id // self.large_small_ratio
        self.free_block_counter[lv0_block_id] += 1
        self.num_free_small_blocks += 1

        prefer_seq_id = self.large_block_to_seq_id[lv0_block_id]
        if prefer_seq_id in self.free_block_id_with_prefer:
            self.free_block_id_with_prefer[prefer_seq_id][block_id] = None
        else:
            self.free_block_id_no_prefer.add(block_id)

        if self.free_block_counter[lv0_block_id] == self.large_small_ratio:
            # all blocks in this large block are free, reture them to large block allocator
            to_free_lv1_block_ids = [
                lv0_block_id * self.large_small_ratio + i
                for i in range(self.large_small_ratio)
            ]
            self.ref_counter.remove_blocks(to_free_lv1_block_ids)
            del self.large_block_to_seq_id[lv0_block_id]
            if prefer_seq_id in self.free_block_id_with_prefer:
                for lv1_block_id in to_free_lv1_block_ids:
                    self.free_block_id_with_prefer[prefer_seq_id].pop(
                        lv1_block_id)
            else:
                self.free_block_id_no_prefer.difference_update(
                    to_free_lv1_block_ids)
            del self.free_block_counter[lv0_block_id]
            self.large_block_id_allocator.free_block_id(lv0_block_id)
            self.num_free_small_blocks -= self.large_small_ratio

    def remove_seq(self, seq_id: int):
        for lv1_block_id in self.free_block_id_with_prefer[seq_id]:
            self.free_block_id_no_prefer.add(lv1_block_id)
        # we do not change large_block_to_seq_id mapping here, instead, we check it when free small blocks
        del self.free_block_id_with_prefer[seq_id]

    def add_seq(self, seq_id: int):
        self.free_block_id_with_prefer[seq_id] = OrderedDict()
