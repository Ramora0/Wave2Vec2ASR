import logging
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers.modeling_outputs import CausalLMOutput
import os

from BoundaryPredictor import BoundaryPredictor

from typing import Optional, Tuple, Union
import torch
from torch import nn
from transformers import Wav2Vec2Model
from transformers.modeling_outputs import Wav2Vec2BaseModelOutput

from BoundaryPredictor import BoundaryPredictor

class MagnetWav2Vec2(Wav2Vec2ForCTC):
    def load_magnet(self, temp, prior, threshold):
        self.wav2vec2.__class__ = MagnetWav2Vec2Model
        self.wav2vec2.load_magnet(temp, prior, threshold)
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, temp=1.0, prior=0.5, threshold=0.5, *model_args, **kwargs):
        # Load the base model using the parent class method
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)

        boundary_params_path = os.path.join(pretrained_model_name_or_path, "boundary_params.pt")
        params = torch.load(boundary_params_path, map_location="cpu")
        temp = params.get("temp", temp)
        prior = params.get("prior", prior)
        threshold = params.get("threshold", threshold)
        
        # Initialize the magnet component with specified parameters
        model.__class__ = MagnetWav2Vec2
        model.load_magnet(temp, prior, threshold)
        
        boundary_path = os.path.join(pretrained_model_name_or_path, "boundary_predictor.bin")
        boundary_state_dict = torch.load(boundary_path, map_location=model.device)
        model.wav2vec2.boundary_predictor.load_state_dict(boundary_state_dict)
        
        return model
    
    def save_pretrained(self, save_directory, *args, **kwargs):
        super().save_pretrained(save_directory, *args, **kwargs)
        
        boundary_path = os.path.join(save_directory, "boundary_predictor.bin")
        torch.save(self.wav2vec2.boundary_predictor.state_dict(), boundary_path)

        boundary_params_path = os.path.join(save_directory, "boundary_params.pt")
        params = {
            "temp": self.wav2vec2.boundary_predictor.temp,
            "prior": self.wav2vec2.boundary_predictor.prior,
            "threshold": self.wav2vec2.boundary_predictor.threshold
        }
        torch.save(params, boundary_params_path)
        
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, target_length)`, *optional*):
            Labels for connectionist temporal classification. Note that `target_length` has to be smaller or equal to
            the sequence length of the output logits. Indices are selected in `[-100, 0, ..., config.vocab_size - 1]`.
            All labels set to `-100` are ignored (masked), the loss is only computed for labels in `[0, ...,
            config.vocab_size - 1]`.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None and labels.max() >= self.config.vocab_size:
            raise ValueError(f"Label values must be <= vocab_size: {self.config.vocab_size}")

        outputs, boundary_loss = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            # retrieve loss input_lengths from attention_mask
            attention_mask = (
                attention_mask if attention_mask is not None else torch.ones_like(input_values, dtype=torch.long)
            )
            # input_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1)).to(torch.long)
            input_lengths = torch.full(size=(logits.shape[0],), fill_value=logits.shape[1], dtype=torch.long, device=logits.device)

            # assuming that padded tokens are filled with -100
            # when not being attended to
            labels_mask = labels >= 0
            target_lengths = labels_mask.sum(-1)
            flattened_targets = labels.masked_select(labels_mask)

            # ctc_loss doesn't support fp16
            log_probs = nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32).transpose(0, 1)

            with torch.backends.cudnn.flags(enabled=False):
                loss = nn.functional.ctc_loss(
                    log_probs,
                    flattened_targets,
                    input_lengths,
                    target_lengths,
                    blank=self.config.pad_token_id,
                    reduction=self.config.ctc_loss_reduction,
                    zero_infinity=self.config.ctc_zero_infinity,
                )

            loss += boundary_loss

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=outputs.hidden_states, attentions=outputs.attentions
        )
    

class MagnetWav2Vec2Model(Wav2Vec2Model):
    def load_magnet(self, temp, prior, threshold):
        self.boundary_predictor = BoundaryPredictor(self.config.conv_dim[-1],
                                                    self.config.conv_dim[-1] // 4,
                                                    temp, prior, threshold)
    
    def forward(
        self,
        input_values: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        mask_time_indices: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Wav2Vec2BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        extract_features = self.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)

        if attention_mask is not None:
            # compute reduced attention_mask corresponding to feature vectors
            attention_mask = self._get_feature_vector_attention_mask(
                extract_features.shape[1], attention_mask, add_adapter=False
            )
            
        # original_seq_len = extract_features.shape[1]
        extract_features, boundary_loss = self.boundary_predictor(extract_features)
        # compressed_seq_len = extract_features.shape[1]
        
        # if self.training and original_seq_len > 0 and compressed_seq_len > 0: # Avoid division by zero and report only during training if desired
        #      compression_ratio = 1 - compressed_seq_len/original_seq_len
        #      with open('boundary_loss.txt', 'a') as f:
        #             f.write(f"{boundary_loss}\n")
            #  print(f"Compression rate: {compression_ratio:.2f} ({original_seq_len} -> {compressed_seq_len}) [{boundary_loss}]")
             

        hidden_states, extract_features = self.feature_projection(extract_features)
        hidden_states = self._mask_hidden_states(
            hidden_states, mask_time_indices=mask_time_indices, attention_mask=attention_mask
        )

        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = encoder_outputs[0]

        if self.adapter is not None:
            hidden_states = self.adapter(hidden_states)

        if not return_dict:
            return (hidden_states, extract_features) + encoder_outputs[1:]

        return Wav2Vec2BaseModelOutput(
            last_hidden_state=hidden_states,
            extract_features=extract_features,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        ), boundary_loss