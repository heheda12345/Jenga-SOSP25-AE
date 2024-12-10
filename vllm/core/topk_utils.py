import math 
from typing import List
from vllm.sequence import Sequence

# NOTE: just use global variable here to store the status 
global cache_topk
global cache_topk_wo_none
cache_topk = {} # seq ID -> its most recent topk for each layer
cache_topk_wo_none = {} # seq ID -> its most recent topk for each layer, with None indiciating non-dropping 

class TopKCalculator:
    def __init__(self, num_layers: int = 80) -> None:
        # NOTE (shu): for token-drop experiments 
        self.prefill_config = {
            "recent_ratio": 0.2,
            "prefill_decay_ratio": 0.9,
            "prefill_decay_strategy": "cosine",
            "min_context_length": 64,
            "layerwise_downsample_interval": 1,
            "streamingllm_sink_len": 4,
            "distance_weight": 1.2
        }
        
        self.gen_config = {
            "gen_decay_ratio": 0.1,
            "gen_decay_strategy": "cosine",
            "gen_compress_ratio": 0.9,
            "exceed_length_to_compress": 16
        }
        
        self.prefill_decay_strategy = self.prefill_config["prefill_decay_strategy"]
        self.prefill_decay_ratio = self.prefill_config["prefill_decay_ratio"]
        self.min_context_length = self.prefill_config["min_context_length"]
        self.layerwise_downsample_interval = self.prefill_config["layerwise_downsample_interval"]
        self.recent_ratio = self.prefill_config["recent_ratio"]
        
        self.gen_decay_strategy = self.gen_config["gen_decay_strategy"]
        self.gen_decay_ratio = self.gen_config["gen_decay_ratio"]
        self.exceed_length_to_compress = self.gen_config["exceed_length_to_compress"]
        self.gen_compress_ratio = self.gen_config["gen_compress_ratio"]
            
        # NOTE (shu): hard code...
        self.num_layers = num_layers
        self.recent_attn_weights_last_shape = {} # seq ID -> its last recent attn weights last shape 
        self.past_kv_seq_lens = {} # seq ID -> its past kv seq lens
        self.past_step_largest_topk = {} # seq ID -> its largest topk in the past step 
        
        # Only for decode layer 
        self.past_largest_topk_all_layers = {} # seq ID -> its largest topk in the past step for each layer (if topk is drop)
        
        # Cache for each seq_id calculated topk if update is True 
        # self.cache_topk = {} # seq ID -> its most recent topk for each layer
    
    ### TODO (shu): calculate synthetic k and update parameter 
    ##################### NOTE(shu): add synthetic k calculation #####################
    def calculate_k_decode(
        self,
        layer_id: int,
        # below original parameters 
        recent_attn_weights_idx_last_shape: List[int],
        past_kv_seq_lens_idx: List[int],
        next_decoder_cache_seen_tokens: int,
        recent_length: int,
        seq_id: int, 
        update: bool = False,
    ) -> None:
        """
        Calculate top K for the decoding stage 
        
        If top K is smaller than the past largest topk, update to the past largest topk
        """
        if self.gen_decay_strategy == "linear":
            schedule_gen_decay_ratio = (1.0 - self.gen_decay_ratio) * (layer_id / self.num_layers) + self.gen_decay_ratio
        if self.gen_decay_strategy == "cosine":
            schedule_gen_decay_ratio = (1.0 - self.gen_decay_ratio) * (math.cos(math.pi * layer_id / self.num_layers) + 1) / 2 + self.gen_decay_ratio
        else:
            schedule_gen_decay_ratio = self.gen_decay_ratio
                            
        last_shape = recent_attn_weights_idx_last_shape + 1    
        past_kv_seq_len = past_kv_seq_lens_idx
        current_kv_seq_len = next_decoder_cache_seen_tokens

        if current_kv_seq_len - recent_length - past_kv_seq_len >= self.exceed_length_to_compress: 
            if 1 + recent_length + self.exceed_length_to_compress > last_shape:
                keep_last_shape = last_shape - (1 + recent_length)
            else:
                keep_last_shape = self.exceed_length_to_compress
            
            context_length = keep_last_shape * self.gen_compress_ratio
            topk = max(int(context_length * schedule_gen_decay_ratio), 1)    
            
            # NOTE: if smaller, update to the past largest topk (that is thrown away)
            # if seq_id in self.past_largest_topk_all_layers and layer_id in self.past_largest_topk_all_layers[seq_id]:
            #         if topk < self.past_largest_topk_all_layers[seq_id][layer_id]:
            #             print(f"Top k {topk} is smaller than the past largest topk {self.past_largest_topk_all_layers[seq_id][layer_id]}")
            #             topk = self.past_largest_topk_all_layers[seq_id][layer_id]
            
            # print(f"DECODE, PREDICT, UPDATE: {update}, SEQ_ID: {seq_id}, LAYER: {layer_id}, TOPK: {topk}")                   
            return topk
        else:
            # print(f"DECODE, PREDICT, UPDATE: {update}, SEQ_ID: {seq_id}, LAYER: {layer_id}, TOPK: None")                   
            return None 
    
    def calculate_k_prefill(
        self,
        seq_id: int,
        layer_id: int,
        prefill_context_len: int,
        num_tokens_candidate: int, 
        update: bool = False,
    ) -> None:
        num_layers = self.num_layers
        if self.prefill_decay_strategy == "linear":
            schedule_prefill_decay_ratio = (1.0 - self.prefill_decay_ratio) * (layer_id / num_layers) + self.prefill_decay_ratio
        if self.prefill_decay_strategy == "cosine":
            schedule_prefill_decay_ratio = (1.0 - self.prefill_decay_ratio) * (math.cos(math.pi * layer_id / num_layers) + 1) / 2 + self.prefill_decay_ratio
        else:
            schedule_prefill_decay_ratio = self.prefill_decay_ratio
            
        num_tokens_candidate = num_tokens_candidate

        topk = None 
        if (layer_id % self.layerwise_downsample_interval) == 0:
            if prefill_context_len > self.min_context_length and schedule_prefill_decay_ratio < 1.0: 
                topk = int(prefill_context_len * schedule_prefill_decay_ratio) if int(prefill_context_len * schedule_prefill_decay_ratio) > self.min_context_length else prefill_context_len
                # seq_len_to_keep = topk + recent_length + 1
                return topk 
            
        # print(f"PREFILL, PREDICT, UPDATE: {update}, SEQ_ID: {seq_id}, LAYER: {layer_id}, TOPK: {topk}")                   
        return topk 
                
    def update_parameter(
        self,
        seq_id,
        layer_id: int, 
        recent_length: int, 
        topk: int, 
    ) -> int:
        """
        Update parameter for recent_attn_weights_last_shape_list() for seq_id at layer_id
        """        
        recent_attn_weights_last_shape_list = self.recent_attn_weights_last_shape[seq_id]
        # print(f"Recent attn weights last shape: {recent_attn_weights_last_shape_list}, recent length: {recent_length}, exceed length to compress: {self.exceed_length_to_compress}")
        
        x = recent_attn_weights_last_shape_list[layer_id] + 1
        
        if x - (1 + recent_length + self.exceed_length_to_compress) < 0:
            first_factor = 0 
        else:
            first_factor = x - (1 + recent_length + self.exceed_length_to_compress)
            
        recent_length_updated_to = first_factor + topk + 1 + recent_length
        return recent_length_updated_to
    
    def get_top_k_for_seq(
        self,
        is_prompt: bool, 
        seq: Sequence,
        update: bool = False, 
    ):
        """
        This function gets top k for each layer for a sequence (seq_id), return as a list.
        It also update seq_id specific lists:
        - recent_attn_weights_last_shape[seq_id]: list of recent_attn_weights_last_shape for each layer
        - past_kv_seq_lens[seq_id]: list of past_kv_seq_lens for each layer

        Args:
            is_prompt (bool): is this for prompt or decode
            seq (Sequence): get top k for this sequence
            update (bool): update the recent_attn_weights_last_shape and past_kv_seq_lens for this seq_id 
                    Only update if this is set to True

        Returns:
            List[int]: list of top k for each layer, with None indiciating non-dropping 
            List[int]: list of top k for each layer, with seq_len indicating non-dropping
            NOTE: this is probably the same for all sequences 
        """
        print_past_kv_seq_lens = self.past_kv_seq_lens[seq.seq_id] if seq.seq_id in self.past_kv_seq_lens else None
        past_recent_weight_list = self.recent_attn_weights_last_shape[seq.seq_id] if seq.seq_id in self.recent_attn_weights_last_shape else None
        
        
        # if update: 
            # print("-"*50)
            # print(f"Past kv seq lens: {print_past_kv_seq_lens}")
            # if past_recent_weight_list is not None:
                # print(f"Recent attn weights (length: {len(past_recent_weight_list)}) last shape:\n {past_recent_weight_list}")
            # else:
            #     print(f"Recent attn weights last shape is None")
            # print("-"*50)
        
        top_k_list = []
        top_k_wo_none_list = []
            
        seq_len = seq.get_len() # TODO (shu): verify (this has to be after run? so shall be +1 or not?)
        inputs_embed_shape_1 = seq_len if is_prompt else 1 # TODO: check 
        seq_id = seq.seq_id
        
        if seq_id not in self.recent_attn_weights_last_shape or seq_id not in self.past_kv_seq_lens:
            if update: 
                self.recent_attn_weights_last_shape[seq_id] = [] 
                self.past_kv_seq_lens[seq_id] = []
            seq_length_with_past = inputs_embed_shape_1 
            # print(f"Just set seq_length_with_past to inputs_embed_shape_1: {inputs_embed_shape_1}")
        else:
            seq_length_with_past = inputs_embed_shape_1 + self.recent_attn_weights_last_shape[seq_id][0]
            # print(f"Seq length with past calculated as input_embed_shape {inputs_embed_shape_1} + last_shape[seq_id][0] {self.recent_attn_weights_last_shape[seq_id][0]}")
            
        # seq_length_with_past = seq_len
            
        recent_length = int(seq_length_with_past * self.recent_ratio)
        # print(f"Sequence length with past: {seq_length_with_past}, recent length: {recent_length}")
        next_decoder_cache_seen_tokens = seq_len 
        past_prefill_length = None 
        past_top_k = None
        
        for layer_id in range(self.num_layers):
            ### Calculate top K for this seq_id and layer_id
            if is_prompt:
                # prefill 
                num_tokens_candidate = seq_len - recent_length - 1 
                prefill_context_len = past_prefill_length if past_prefill_length is not None else num_tokens_candidate
                
                topk = self.calculate_k_prefill(
                    seq_id,
                    layer_id,
                    prefill_context_len,
                    num_tokens_candidate=num_tokens_candidate,
                )
            
                # UPDATE past_kv_seq_lens and recent_attn_weights_last_shape for this seq_id
                if update: 
                    if past_top_k is None:
                        self.past_kv_seq_lens[seq_id].append(num_tokens_candidate + recent_length + 1)
                    else:
                        self.past_kv_seq_lens[seq_id].append(past_top_k + recent_length + 1)
                        
                    self.recent_attn_weights_last_shape[seq_id].append(self.past_kv_seq_lens[seq_id][-1])
                
                if past_top_k is None:
                    past_prefill_length = num_tokens_candidate
                else:
                    past_prefill_length = topk
                    
                past_top_k = topk if topk is not None else num_tokens_candidate  
                
                # For prompt, append topk to the list (never gonna be None)    
                top_k_list.append(topk)
                # print(f"Top k now seq len is: {seq_len}")
                top_k_wo_none_list.append(seq_len)
                # top_k_wo_none_list.append(past_top_k)
            else:
                # decode 
                assert seq_id in self.recent_attn_weights_last_shape, f"Seq id: {seq_id} not in recent_attn_weights_last_shape"
                assert seq_id in self.past_kv_seq_lens, f"Seq id: {seq_id} not in past_kv_seq_lens"
                assert len(self.recent_attn_weights_last_shape[seq_id]) > 0, f"Seq id: {seq_id} recent_attn_weights_last_shape is empty"
                assert len(self.past_kv_seq_lens[seq_id]) > 0, f"Seq id: {seq_id} past_kv_seq_lens is empty"
                
                predicted_top_k = self.calculate_k_decode(
                    layer_id,
                    self.recent_attn_weights_last_shape[seq_id][layer_id],
                    self.past_kv_seq_lens[seq_id][layer_id],
                    next_decoder_cache_seen_tokens, 
                    recent_length, 
                    seq_id,
                    update=update,
                )      

                if update: 
                    # UPDATE past_kv_seq_lens and recent_attn_weights_last_shape for this seq_id 
                    if predicted_top_k is not None:
                        # Update topK related calculation 
                        self.recent_attn_weights_last_shape[seq_id][layer_id] = self.update_parameter(
                            seq_id, layer_id, recent_length, predicted_top_k)
                        
                        self.past_kv_seq_lens[seq_id][layer_id] = self.recent_attn_weights_last_shape[seq_id][layer_id] - recent_length

                    else:                
                        # Update topK related calculation 
                        if len(self.recent_attn_weights_last_shape[seq_id]) > 0:
                            self.recent_attn_weights_last_shape[seq_id][layer_id] += 1 
                        
                        
                # NOTE: seq_len is prompt + decode if not throw away 
                # NOTE: is this correct?
                append_top_k = predicted_top_k if predicted_top_k is not None else seq_len 
                top_k_list.append(predicted_top_k)
                top_k_wo_none_list.append(append_top_k)
        
        if update:
            # Logic 
            # past req-level, last step largest top k is still kept with the top_k_wo_none_list 
            # when assigning top k list, for those with not None 
            # - if token drop k < previous largest k of this seq_id in this layer, then update to the previous largest k
            # - keep track of the largest k for this seq_id for each of the layer (if token drop)
            max_tokens_keep_all_layers = max(top_k_wo_none_list)
            self.past_step_largest_topk[seq_id] = max_tokens_keep_all_layers
            
            
            # print(f"Top k list for seq id {seq_id}: {len(cache_topk[seq_id])}")
            global cache_topk
            cache_topk[seq_id] = top_k_list # cache most recent topk for each layer, if None, keep all
            
            global cache_topk_wo_none
            cache_topk_wo_none[seq_id] = top_k_wo_none_list
            
        return top_k_list, top_k_wo_none_list

        
    def get_past_step_largest_topk(self, seq_id: int) -> int:
        """
        Return the largest topk for this seq_id in the past step
        """
        return self.past_step_largest_topk[seq_id]
    
    # def get_all_prev_largest_topk(self, seq_id: int) -> int:
    #     """
    #     Return the largest topk for this seq_id among all the past steps
    #     """
    #     # return self.overall_past_largest_topk[seq_id]   
    #     return self.overall_past_largest_topk[seq_id]
    
    
