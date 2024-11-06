from collections import deque

from vllm.utils import Device


class LargeBlockIDAllocator:

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
        return self.free_block_ids.popleft()

    def free_block_id(self, block_id):
        self.free_block_ids.append(block_id)

    def clear_copy_on_writes(self):
        return []
