from transformers.models.llama import LlamaConfig


class CharacterConfig(LlamaConfig):
    '''
    Use this class to override is_encoder_decoder:
    - transformers regards mllama as is_encoder_decoder=False
    - vllm needs is_encoder_decoder=True to enable cross-attention
    '''

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)
        print("kwargs", kwargs)
        self.kv_share = kwargs.get('kv_share', None)
        if '_sliding_window' in kwargs:
            self._sliding_window = tuple(kwargs.get('_sliding_window'))
        else:
            self._sliding_window = None
        # leave self.sliding_window as None
        self.model_type = 'character'
