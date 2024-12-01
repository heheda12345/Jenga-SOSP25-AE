from transformers.models.llama import LlamaConfig


class CharacterConfig(LlamaConfig):
    '''
    Use this class to override is_encoder_decoder:
    - transformers regards mllama as is_encoder_decoder=False
    - vllm needs is_encoder_decoder=True to enable cross-attention
    '''

    def __init__(
        self,
        _sliding_window=None,
        kv_share=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.kv_share = kv_share
        if _sliding_window is not None:
            self._sliding_window = tuple(_sliding_window)
        else:
            self._sliding_window = None
        # leave self.sliding_window as None
        self.model_type = 'character'
