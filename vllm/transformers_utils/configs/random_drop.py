from transformers.models.llama import LlamaConfig


class RandomDropConfig(LlamaConfig):
    '''
    Use LlamaConfig as a base class for RandomDropConfig
    NOTE: might need to modify something here
    '''
    def __init__(
        self,
        **kwargs,
    ):
        print("kwargs for config init", kwargs)
        import traceback
        traceback.print_stack()

        super().__init__(**kwargs)
        self.model_type = 'random_drop'
        
    # def __init__(
    #     self,
    #     _sliding_window=None,
    #     kv_share=None,
    #     **kwargs,
    # ):
    #     print("kwargs for config init", kwargs)
    #     import traceback
    #     traceback.print_stack()
    #     super().__init__(**kwargs)
    #     self.kv_share = kv_share
    #     if _sliding_window is not None:
    #         self._sliding_window = tuple(_sliding_window)
    #     else:
    #         self._sliding_window = None
    #     # leave self.sliding_window as None
    #     self.model_type = 'character'
