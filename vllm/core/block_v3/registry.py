from abc import abstractmethod
from typing import Protocol, Dict, Any, Type, TYPE_CHECKING
from torch import nn
from typing_extensions import TypeVar
from vllm.config import CacheConfig, ParallelConfig, SchedulerConfig
from vllm.core.block.block_table import BlockTable
from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import Device, cdiv, chunk_list
from vllm.logger import init_logger
from vllm.config import ModelConfig
from vllm.core.block_v3.custom_block import AppAwareAttnMetadataBuilder

logger = init_logger(__name__)


class BlockTableFactory(Protocol):

    def __call__(self, model_config: ModelConfig) -> Dict[Any, BlockTable]:
        ...


N = TypeVar("N", bound=Type[nn.Module])


class BlockManagerRegistry:

    def __init__(self) -> None:
        self._block_manager_factories_by_model_type: Dict[
            Type[nn.Module], BlockTableFactory] = {}

    def register_block_manager(self, factory: BlockTableFactory):
        # TODO: only support block_manager_v3

        def wrapper(model_cls: N) -> N:
            if model_cls in self._block_manager_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._block_manager_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def get_managers_of_model(
        self, model_config: ModelConfig, cache_config: CacheConfig,
        parallel_config: ParallelConfig, scheduler_config: SchedulerConfig
    ) -> Dict[Any, "AppAwareAttnMetadataBuilder"]:
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        custom_block_manager_func = self._block_manager_factories_by_model_type \
           .get(model_cls, self._default_block_table_factory)
        return custom_block_manager_func(model_config, cache_config,
                                         parallel_config, scheduler_config)

    def _default_block_table_factory(self, model_config: ModelConfig):
        """
        The default block table factory represents the longest possible text
        that can be inputted to the model.
        """
        raise NotImplementedError(
            "TODO: Implement default block table factory")

    def support_model(self, model_config: ModelConfig) -> bool:
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        return model_cls in self._block_manager_factories_by_model_type


BLOCK_MANAGER_REGISTRY = BlockManagerRegistry()
