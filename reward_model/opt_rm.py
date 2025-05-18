import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from tqdm import tqdm

from transformers import OPTForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

import logging

class MyOPTForCausalLM(OPTForCausalLM):
    def __init__(self, config, forward_mode="reward_train"):
        super().__init__(config)
        self.model_name = "MyOPTForCausalLM"        
        
        self.score = torch.nn.Sequential(
            torch.nn.Linear(config.word_embed_proj_dim, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 1)
        )
       
        self.set_forward_mode(forward_mode)

    def set_forward_mode(self, forward_mode="reward_train"):
        ## forward_mode: "base", "reward_train", "reward_guide"
        self.forward_mode = forward_mode
        if forward_mode == "reward_guide":
            self.score = self.score.to(dtype=torch.float32)
    
    
    def set_score_model(self, score_path):
        state_dict = torch.load(score_path)
        self.score.load_state_dict(state_dict)
        print(f"load score model from {score_path}")
        print(f"score state_dict is {state_dict.keys()}")
        self.score = self.score.to(self.model.dtype)

    def set_lr_and_epochs(self, lr, epochs):
        self.guide_lr = lr
        self.guide_epochs = epochs
        self.optimize_log = True


    def set_reward_direction(self, pos_mean, neg_mean, direction):
        self.pos_mean = torch.tensor(pos_mean) 
        self.neg_mean = torch.tensor(neg_mean) 
        self.direction = direction
        self.direction = self.direction.to(device=next(self.score.parameters()).device)

        logging.info(f"pos_mean: {self.pos_mean}, neg_mean: {self.neg_mean}")
        logging.info(f"direction.shape: {self.direction.shape}")


    def set_train_score(self):
        for param in self.model.parameters():
            param.requires_grad = False
        
        for param in self.lm_head.parameters():
            param.requires_grad = False

        for param in self.score.parameters():
            param.requires_grad = True

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_index: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        
        if self.forward_mode == "base":
            return self.forward_base(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        elif self.forward_mode == "reward_train" or self.forward_mode == "reward_eval":
            return self.forward_reward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif self.forward_mode == "reward_guide":
            return self.forward_guide(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                start_index=start_index,
            )
    def forward_base(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = self.lm_head(outputs[0]).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def forward_guide(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        start_index: Optional[int] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]

        
        if start_index is not None:
            hidden_states[:, start_index:, :] = self.optimize_hidden_states_with_reward(hidden_states[:,start_index:,:].detach().clone())
        else:
            hidden_states[:,-1:,:] = self.optimize_hidden_states_with_reward(hidden_states[:,-1:,:].detach().clone())
       
        logits = self.lm_head(hidden_states).contiguous()

        loss = None
        if labels is not None:
            # move labels to correct device to enable model parallelism
            labels = labels.to(logits.device)
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    
    def optimize_hidden_states_with_reward(self, hidden_states):
        
        raw_decive = hidden_states.device
        raw_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(dtype=torch.float32)
        
        
        reward = self.score(hidden_states.detach().clone())         
        reward_gap = self.pos_mean.to(dtype=reward.dtype, device=reward.device) - reward
        
        mask = (reward_gap > 0).float() 
        logging.info(f"mask: {mask.shape}") 
        adjustment = self.direction * reward_gap * mask
        hidden_states_opt = hidden_states.detach().clone() + adjustment.to(device=raw_decive)
        
        
        hidden_states_opt.requires_grad = True
        hidden_states_opt = hidden_states_opt.to(device=next(self.score.parameters()).device)
        
        optimizer = torch.optim.SGD([hidden_states_opt], lr=self.guide_lr)  

        with torch.enable_grad():
            for iteration in range(self.guide_epochs):  
                optimizer.zero_grad()  
                reward = self.score(hidden_states_opt)    
                
                logging.info(f"Epoch {iteration}, reward: {reward.mean().item()}")

                loss = -reward.mean() 
                loss.backward()
                optimizer.step()  
                if self.optimize_log:
                    logging.info(f"Epoch {iteration}, reward: {reward.mean().item()}, loss: {loss.item()}")
                print(f"Epoch {iteration}, reward: {reward.mean().item()}, loss: {loss.item()}")

        self.optimize_log = False        
        hidden_states = hidden_states_opt.detach().clone()  
        hidden_states = hidden_states.to(device=raw_decive, dtype=raw_dtype)

        return hidden_states
    

    def forward_reward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        token_type_ids: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for language modeling. Note that the labels **are shifted** inside the model, i.e. you can set
            `labels = input_ids` Indices are selected in `[-100, 0, ..., config.vocab_size]` All labels set to `-100`
            are ignored (masked), the loss is only computed for labels in `[0, ..., config.vocab_size]`
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            head_mask=head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        
        lm_logits = self.score(hidden_states)


        loss = None
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
