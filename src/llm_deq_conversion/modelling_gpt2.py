import torch
from torchdeq.solver import get_solver, simple_fixed_point_iter
import torch.nn as nn
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.gpt2.modeling_gpt2 import GPT2Model, GPT2LMHeadModel
from transformers.models.gpt2.configuration_gpt2 import GPT2Config
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions
)
from transformers.utils import (
    logging,
)
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa
from typing import Optional, Tuple, Union
from dataclasses import dataclass


logger = logging.get_logger(__name__)

@dataclass
class DEQBaseModelOutputWithPastAndCrossAttentions(BaseModelOutputWithPastAndCrossAttentions):
    stats: Optional[dict[str, torch.Tensor]] = None

    
@dataclass
class DEQCausalLMOutputWithCrossAttentions(CausalLMOutputWithCrossAttentions):
    stats: Optional[dict[str, torch.Tensor]] = None


class DEQGPT2Model(GPT2Model):
    def __init__(self, config: GPT2Config, deq_steps: int = 4, phantom_steps: int = 1, damp: float = 0.9, solver: str = "fixed_point_iter", return_final: bool = True):
        super().__init__(config)
        self.phantom_steps = phantom_steps
        self.deq_steps = deq_steps
        self.damp = damp
        self.return_final = return_final
        # self.adapter = nn.Linear(2*config.hidden_size, config.hidden_size)
        self.solver = get_solver(solver)
    
    def _run_blocks(
        self,
        input_embeds: torch.Tensor,
        hidden_states: torch.Tensor,
        past_key_values: Optional[Union[tuple[tuple[torch.Tensor]], Cache]],
        cache_position: torch.LongTensor,
        causal_mask: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor],
        head_mask: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.Tensor],
        encoder_attention_mask: Optional[torch.FloatTensor],
        use_cache: bool,
        output_attentions: bool,
        output_hidden_states: bool,
        **kwargs,
    ) -> Union[torch.Tensor, Optional[tuple[torch.tensor]], Optional[tuple[torch.tensor]], Optional[tuple[torch.tensor]]]:
        # hidden_states = self.adapter(
            # torch.concat([input_embeds, hidden_states.reshape(input_embeds.shape)])
        # )
        hidden_states = input_embeds + hidden_states.reshape_as(input_embeds)
        # Runs through all transformer blocks and collects outputs
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None
        all_hidden_states = () if output_hidden_states else None
        for i, block in enumerate(self.h):
            # Model parallel handling
            if self.model_parallel:
                torch.cuda.set_device(hidden_states.device)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(hidden_states.device)
                if isinstance(head_mask, torch.Tensor):
                    head_mask = head_mask.to(hidden_states.device)
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            outputs = block(
                hidden_states,
                past_key_values if not (self.gradient_checkpointing and self.training) else None,
                cache_position,
                causal_mask,
                head_mask[i],
                encoder_hidden_states,  # gradient checkpointing pos arg
                encoder_attention_mask=encoder_attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                **kwargs,
            )
            hidden_states = outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (outputs[2],)

            # Model Parallel next device transfer
            if self.model_parallel:
                for k, v in self.device_map.items():
                    if i == v[-1] and "cuda:" + str(k) != self.last_device:
                        hidden_states = hidden_states.to("cuda:" + str(k + 1))

        # Final layer normalization
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.reshape((-1, hidden_states.shape[-1]))
        return hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[tuple[tuple[torch.Tensor]], Cache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        deq_steps: Optional[int] = None,
        phantom_steps: Optional[int] = None,
        **kwargs,
    ) -> Union[tuple, BaseModelOutputWithPastAndCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        """
        deq_steps = deq_steps or self.deq_steps
        phantom_steps = phantom_steps or self.phantom_steps
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache: bool = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            self.warn_if_padding_and_no_attention_mask(input_ids, attention_mask)
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
            batch_size = input_ids.shape[0]
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size = inputs_embeds.shape[0]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if token_type_ids is not None:
            token_type_ids = token_type_ids.view(-1, input_shape[-1])

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        # based on pattern from src/transformers/models/whisper/modeling_whisper.py::WhisperDecoder
        return_legacy_cache = False
        if use_cache:
            if past_key_values is None:
                return_legacy_cache = True
                past_key_values = DynamicCache()
            elif not isinstance(past_key_values, Cache):
                return_legacy_cache = True
                logger.warning_once(
                    "Passing a tuple of `past_key_values` is deprecated and will be removed in Transformers v4.53.0. "
                    "You should pass an instance of `Cache` instead, e.g. "
                    "`past_key_values=DynamicCache.from_legacy_cache(past_key_values)`."
                )
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)

            if self.config.add_cross_attention and not isinstance(past_key_values, EncoderDecoderCache):
                past_key_values = EncoderDecoderCache(past_key_values, DynamicCache())

        if inputs_embeds is None:
            inputs_embeds = self.wte(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        position_embeds = self.wpe(position_ids)
        inputs_embeds = inputs_embeds + position_embeds.to(inputs_embeds.device)

        # Attention mask.
        # ._update_causal_mask() and ._prepare_4d_causal_attention_mask_with_cache_position() copied from LlamaModel
        if attention_mask is not None and attention_mask.ndim < 4:
            attention_mask = attention_mask.view(batch_size, -1)
        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        _use_sdpa = self._attn_implementation == "sdpa" and output_attentions is False and head_mask is None
        if self.config.add_cross_attention and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            if _use_sdpa:
                encoder_attention_mask = _prepare_4d_attention_mask_for_sdpa(
                    mask=encoder_attention_mask, dtype=inputs_embeds.dtype, tgt_len=input_shape[-1]
                )
            elif not self._attn_implementation == "flash_attention_2":
                encoder_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        head_mask = self.get_head_mask(head_mask, self.config.n_layer)

        if token_type_ids is not None:
            token_type_embeds = self.wte(token_type_ids)
            inputs_embeds = inputs_embeds + token_type_embeds

        inputs_embeds = self.drop(inputs_embeds)
        hidden_states = torch.zeros_like(inputs_embeds).reshape(-1, inputs_embeds.shape[-1])
        output_shape = (-1,) + input_shape[1:] + (inputs_embeds.size(-1),)

        deq_forward = lambda x, tau=1.0: self._run_blocks(
            input_embeds=inputs_embeds,
            hidden_states=x,
            past_key_values=past_key_values,
            cache_position=cache_position,
            causal_mask=causal_mask,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states
        )[0] * tau + (1 - tau) * x
        stats = None
        if deq_steps > 0:
            with torch.no_grad():
                hidden_states, _, stats = self.solver(
                    func=deq_forward,
                    x0=hidden_states,
                    max_iter=deq_steps,
                    tau=1.0,
                    stop_mode='rel',
                    return_final=self.return_final, 
                )
                
        if phantom_steps > 0:
            hidden_states, _, stats = simple_fixed_point_iter(
                func=deq_forward,
                x0=hidden_states,
                max_iter=phantom_steps,
                tau=self.damp,
            )

        hidden_states = hidden_states.view(output_shape)
        # Add last hidden state
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        past_key_values = past_key_values if use_cache else None
        if return_legacy_cache:
            past_key_values = (
                past_key_values.self_attention_cache.to_legacy_cache()
                if self.config.add_cross_attention
                else past_key_values.to_legacy_cache()
            )
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, past_key_values, all_hidden_states, all_self_attentions, all_cross_attentions]
                if v is not None
            )

        return DEQBaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            stats=stats
        )

    
class DEQGPT2LMHeadModel(GPT2LMHeadModel):
    def __init__(self, config: GPT2Config, deq_steps: int = 4, phantom_steps: int = 1, damp: float = 0.9, solver: str = "fixed_point_iter", return_final: bool = True):
        super().__init__(config)
        self.transformer = DEQGPT2Model(config, deq_steps=deq_steps, phantom_steps=phantom_steps, damp=damp, solver=solver, return_final=return_final)
        self.post_init()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, DEQCausalLMOutputWithCrossAttentions]:
        r"""
        input_ids (`torch.LongTensor` of shape `(batch_size, input_ids_length)`):
            `input_ids_length` = `sequence_length` if `past_key_values` is `None` else
            `past_key_values[0][0].shape[-2]` (`sequence_length` of input past key value states). Indices of input
            sequence tokens in the vocabulary.

            If `past_key_values` is used, only `input_ids` that do not have their past calculated should be passed as
            `input_ids`.

            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.

            [What are input IDs?](../glossary#input-ids)
        labels (`torch.LongTensor` of shape `(batch_size, input_ids_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        transformer_outputs = self.transformer(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = transformer_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.transformer.first_device)
            hidden_states = hidden_states.to(self.lm_head.weight.device)

        lm_logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # Flatten the tokens
            loss = self.loss_function(
                lm_logits,
                labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = (lm_logits,) + transformer_outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return DEQCausalLMOutputWithCrossAttentions(
            loss=loss,
            logits=lm_logits,
            stats=transformer_outputs.stats,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
            cross_attentions=transformer_outputs.cross_attentions,
        )
        
        
if __name__ == "__main__":
    from transformers import AutoTokenizer
    config = GPT2Config.from_pretrained("openai-community/gpt2")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # load state dict from huggingface
    original_model = GPT2LMHeadModel.from_pretrained("openai-community/gpt2", config=config)
    original_model.eval()
    
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    model = DEQGPT2LMHeadModel(config, max_steps=0, phantom_steps=1, damp=1.0, solver="fixed_point_iter", return_final=True)
    model.eval()
    errors = model.load_state_dict(original_model.state_dict(), strict=False)
    print(errors)
    train_examples = {
        "input_ids": inputs["input_ids"][:, :-1],
        "attention_mask": inputs["attention_mask"][:, :-1],
        "labels": inputs["input_ids"][:, 1:],
    }
    outputs = model(**train_examples, use_cache=False)
    origin_outputs = original_model(**train_examples, use_cache=False)
    print("Are the outputs equal?", torch.allclose(outputs.loss, origin_outputs.loss, atol=1e-4))
