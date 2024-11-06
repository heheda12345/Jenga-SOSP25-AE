from typing import List, Optional
from vllm.core.block.interfaces import Block, BlockId


class NullBlock(Block):
    """
    Null blocks are used as a placeholders for KV cache blocks that have
    been dropped due to sliding window.
    This implementation just wraps an ordinary block and prevents it from
    being modified. It also allows for testing if a block is NullBlock
    via isinstance().
    """

    def __init__(self, proxy: Block):
        super().__init__()
        self._proxy = proxy

    def append_token_ids(self, token_ids: List[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def block_id(self):
        return self._proxy.block_id

    @block_id.setter
    def block_id(self, value: Optional[BlockId]):
        raise ValueError("null block should not be modified")

    @property
    def token_ids(self) -> List[BlockId]:
        return self._proxy.token_ids

    @property
    def num_tokens_total(self) -> int:
        raise NotImplementedError(
            "num_tokens_total is not used for null block")

    @property
    def num_empty_slots(self) -> BlockId:
        return self._proxy.num_empty_slots

    @property
    def is_full(self):
        return self._proxy.is_full

    @property
    def prev_block(self):
        return self._proxy.prev_block

    @property
    def computed(self):
        return self._proxy.computed

    @computed.setter
    def computed(self, value):
        self._proxy.computed = value

    @property
    def last_accessed(self) -> float:
        return self._proxy.last_accessed

    @last_accessed.setter
    def last_accessed(self, last_accessed_ts: float):
        self._proxy.last_accessed = last_accessed_ts

    @property
    def content_hash(self):
        return self._proxy.content_hash


NULL_BLOCK_SEQ_ID = -10
