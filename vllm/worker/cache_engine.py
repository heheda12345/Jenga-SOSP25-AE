"""CacheEngine class for managing the KV cache."""
from functools import reduce
import operator
from typing import Dict, List, Optional

import torch

from vllm.attention import get_attn_backend
from vllm.config import CacheConfig, DeviceConfig, KVCacheConfig, KVPageType, ModelConfig, ParallelConfig
from vllm.logger import init_logger
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, TORCH_DTYPE_TO_STR_DTYPE,
                        get_dtype_size, is_pin_memory_available)

logger = init_logger(__name__)


class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(self, cache_config: CacheConfig, model_config: ModelConfig,
                 parallel_config: ParallelConfig, device_config: DeviceConfig,
                 use_per_layer_attn_metadata: bool) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config

        self.head_size = model_config.get_head_size()
        # Models like Jamba, have mixed typed layers, E.g Mamba
        self.num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)

        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks:
            self.num_gpu_blocks //= parallel_config.pipeline_parallel_size
        self.num_cpu_blocks = cache_config.num_cpu_blocks
        if self.num_cpu_blocks:
            self.num_cpu_blocks //= parallel_config.pipeline_parallel_size

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        self.use_per_layer_attn_metadata = use_per_layer_attn_metadata

        # Get attention backend.
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            self.head_size,
            self.num_kv_heads,
            model_config.get_sliding_window(),
            model_config.dtype,
            cache_config.cache_dtype,
            self.block_size,
        )

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, self.device_config.device_type)
        self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    def _allocate_kv_cache(self, num_blocks: int,
                           device: str) -> List[torch.Tensor]:
        """Allocates KV cache on the specified device."""
        kv_cache_shape = self.attn_backend.get_kv_cache_shape(
            num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: List[torch.Tensor] = []
        for _ in range(self.num_attention_layers):
            # null block in CpuGpuBlockAllocator requires at least that
            # block to be zeroed-out.
            # We zero-out everything for simplicity.
            kv_cache.append(
                torch.zeros(kv_cache_shape,
                            dtype=self.dtype,
                            pin_memory=pin_memory,
                            device=device))
        if self.use_per_layer_attn_metadata:
            kv_cache = {str(i): c for i, c in enumerate(kv_cache)}
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        head_size = model_config.get_head_size()
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_attention_layers(
            parallel_config)

        key_cache_block = cache_config.block_size * num_heads * head_size
        value_cache_block = key_cache_block
        total = num_attention_layers * (key_cache_block + value_cache_block)
        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total


class CacheEngineV3:

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.kv_cache_config = kv_cache_config

        self.head_size = model_config.get_head_size()
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.block_size = cache_config.block_size

        # Get attention backend.
        assert model_config.dtype == kv_cache_config.buffer_dtype
        self.attn_backend = get_attn_backend(
            model_config.get_num_attention_heads(parallel_config),
            model_config.get_head_size(),
            model_config.get_num_kv_heads(parallel_config),
            model_config.get_sliding_window(),
            model_config.dtype,
            TORCH_DTYPE_TO_STR_DTYPE[kv_cache_config.buffer_dtype],
            self.block_size,
        )

        if cache_config.cache_dtype == "auto":
            self.dtype = model_config.dtype
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.device_config.device_type)
        self.cpu_cache = None  # we do not need CPU cache in this prototype

    def get_per_layer_block_table(self, block_table: dict[str, torch.Tensor]):
        raise NotImplementedError

    def _allocate_kv_cache(
        self,
        device: str,
    ) -> Dict[str, torch.Tensor]:
        """Allocates KV cache on the specified device."""
        print("============CALL ALLOCATE KV CACHE")
        kv_cache_layout = self.attn_backend.get_kv_cache_layout(
            self.block_size, self.num_kv_heads, self.head_size)
        self.page_is_leading_dim = self.attn_backend.page_is_leading_dim()
        pin_memory = is_pin_memory_available() if device == "cpu" else False
        kv_cache: Dict[str, torch.Tensor] = {}
        torch.cuda.synchronize()
        free_gpu_memory, total_gpu_memory = torch.cuda.mem_get_info()
        logger.info("Free GPU memory: %d GB, Total GPU memory: %d GB",
                    free_gpu_memory / 1024 / 1024 / 1024,
                    total_gpu_memory / 1024 / 1024 / 1024)
        buffer = torch.zeros((self.kv_cache_config.buffer_size, ),
                             dtype=self.dtype,
                             pin_memory=pin_memory,
                             device=device)
        torch.cuda.synchronize()
        for layer_id, component in self.kv_cache_config.components.items():
            if isinstance(component, KVPageType):
                assert component.page_size == reduce(operator.mul,
                                                     kv_cache_layout) * 2
                half_page_size = component.page_size // 2
                assert half_page_size % 2 == 0
                assert component.num_elements % component.page_size == 0
                if self.page_is_leading_dim:
                    shape = (2, component.num_elements // component.page_size,
                             *kv_cache_layout)
                    kv_cache[layer_id] = buffer[
                        component.start_bias:component.start_bias +
                        component.num_elements].view(shape)
                else:
                    shape = (component.num_elements // half_page_size,
                             *kv_cache_layout)
                    k_cache = buffer[component.
                                     start_bias:component.start_bias +
                                     component.num_elements].view(shape)
                    v_cache = buffer[component.start_bias +
                                     half_page_size:component.start_bias +
                                     half_page_size +
                                     component.num_elements].view(shape)
                    kv_cache[layer_id] = (k_cache, v_cache)
            else:
                kv_cache[layer_id] = buffer[component.start_bias:component.
                                            end_bias]
        return kv_cache

    def swap_in(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
                                          src_to_dst)

    def swap_out(self, src_to_dst: torch.Tensor) -> None:
        raise NotImplementedError
        for i in range(self.num_attention_layers):
            self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
                                          src_to_dst)

    def copy(self, src_to_dsts: torch.Tensor) -> None:
        raise NotImplementedError
        self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)

    # should be removed in tue future
    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
    ) -> int:
        raise NotImplementedError
