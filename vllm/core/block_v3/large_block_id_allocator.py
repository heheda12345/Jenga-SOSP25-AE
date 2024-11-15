from collections import deque
from typing import Optional

from vllm.utils import Device


class LargeBlockIDAllocator:

    class NoFreeLevel0BlocksError(ValueError):
        pass

    def __init__(self, num_blocks: int):
        self.num_blocks = num_blocks
        self.free_block_ids = deque(range(num_blocks))

    def get_num_free_blocks(self, device=Device.GPU):
        if device == Device.GPU:
            return len(self.free_block_ids)
        elif device == Device.CPU:
            return 0
        else:
            raise ValueError(f"Unsupported device: {device}")

    def allocate_block_id(self):
        if len(self.free_block_ids) == 0:
            raise self.NoFreeLevel0BlocksError()
        return self.free_block_ids.popleft()

    def free_block_id(self, block_id):
        self.free_block_ids.append(block_id)

    def clear_copy_on_writes(self):
        return []

    def get_prefix_cache_hit_rate(self,
                                  device: Optional[Device] = None) -> float:
        return -1
