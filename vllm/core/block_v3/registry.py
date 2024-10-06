from abc import abstractmethod
from typing import Protocol, Dict, Any, Type, TYPE_CHECKING
from torch import nn
from typing_extensions import TypeVar
from vllm.core.block.block_table import BlockTable
from vllm.sequence import Sequence, SequenceGroup
from vllm.utils import Device, cdiv, chunk_list
from vllm.logger import init_logger
if TYPE_CHECKING:
    from vllm.config import ModelConfig
    from vllm.core.block_v3.custom_block_manager import CustomBlockManager

logger = init_logger(__name__)


class BlockTableFactory(Protocol):

    def __call__(self, model_config: "ModelConfig") -> Dict[Any, "BlockTable"]:
        ...


N = TypeVar("N", bound=Type[nn.Module])


class BlockTableRegistry:

    def __init__(self) -> None:
        self._block_table_factories_by_model_type: Dict[
            Type[nn.Module], BlockTableFactory] = {}

    def register_block_table(self, factory: BlockTableFactory):
        # TODO: only support block_manager_v3

        def wrapper(model_cls: N) -> N:
            if model_cls in self._block_table_factories_by_model_type:
                logger.warning(
                    "Model class %s already has dummy data "
                    "registered to %s. It is overwritten by the new one.",
                    model_cls, self)

            self._block_table_factories_by_model_type[model_cls] = factory

            return model_cls

        return wrapper

    def add_to_block_manager(self, model_config: "ModelConfig",
                             custom_manager: "CustomBlockManager") -> None:
        # Avoid circular import
        from vllm.model_executor.model_loader import get_model_architecture

        model_cls, _ = get_model_architecture(model_config)
        custom_block_table_func = self._block_table_factories_by_model_type \
           .get(model_cls, self._default_block_table_factory)
        custom_manager.add_block_tables(custom_block_table_func(model_config))

    def _default_block_table_factory(self, model_config: "ModelConfig"):
        """
        The default block table factory represents the longest possible text
        that can be inputted to the model.
        """
        raise NotImplementedError(
            "TODO: Implement default block table factory")


BLOCK_TABLE_REGISTRY = BlockTableRegistry()
