from transformers import (AdamW, T5Tokenizer, MT5ForConditionalGeneration, WEIGHTS_NAME, CONFIG_NAME)
from copy import deepcopy
import torch
from torch.nn import CrossEntropyLoss
import time

from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput


class MiniT5(MT5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # make a copy of decoder for dst
        decoder_config = deepcopy(config)
        decoder_config.is_decoder = True

        self.dst_decoder = type(self.decoder)(decoder_config, self.shared)
        self.dst_decoder.load_state_dict(self.decoder.state_dict())
        self.dst_lm_head = type(self.lm_head)(config.d_model, config.vocab_size, bias=False)
        self.dst_lm_head.load_state_dict(self.lm_head.state_dict())

    def tie_decoder(self):
        decoder_config = deepcopy(self.config)
        decoder_config.is_decoder = True
        self.dst_decoder = type(self.decoder)(decoder_config, self.shared)
        self.dst_decoder.load_state_dict(self.decoder.state_dict())
        self.dst_lm_head = type(self.lm_head)(self.config.d_model, self.config.vocab_size, bias=False)
        self.dst_lm_head.load_state_dict(self.lm_head.state_dict())

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        lm_labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        if decoder_input_ids[0, 0] == self.config.decoder_start_token_id:
            decoder = self.dst_decoder
            lm_head = self.dst_lm_head
        else:
            decoder = self.decoder
            lm_head = self.lm_head

        # Encode if needed (training, first prediction pass)
        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                head_mask=head_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        hidden_states = encoder_outputs[0]

        if lm_labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(lm_labels)

        # Decode
        decoder_outputs = decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=past_key_values,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if lm_labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), lm_labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not return_dict:
            output = (lm_logits,) + decoder_outputs[1:] + encoder_outputs
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )

    def inference(
            self,
            tokenizer,
            reader,
            prev,
            input_ids=None,
            attention_mask=None,
            turn_domain=None,
            db=None
    ):
        # start = time.time()
        dst_outputs = self.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    eos_token_id=tokenizer.encode("<eos_b>")[0],
                                    decoder_start_token_id=self.config.decoder_start_token_id,
                                    max_length=200,
                                    )
        # dst_time = time.time()-start
        # print(dst_time)
        dst_outputs = dst_outputs.tolist()
        # length = len(dst_outputs[0])
        # print(dst_outputs)
        # DST_UPDATE -> DST
        # check whether need to add eos
        # dst_outputs = [dst+tokenizer.encode("<eos_b>") for dst in dst_outputs]
        batch_size = input_ids.shape[0]
        constraint_dict_updates = [reader.bspan_to_constraint_dict(tokenizer.decode(dst_outputs[i])) for i in
                                   range(batch_size)]

        if prev['bspn']:
            # update the belief state
            dst_outputs = [reader.update_bspn(prev_bspn=prev['bspn'][i], bspn_update=dst_outputs[i]) for i in
                           range(batch_size)]

        # compute the DB state using the updated domain
        db_state = []
        for bi, bspn_list in enumerate(dst_outputs):
            # if not constraint_dict_updates[bi]:
            #     # if nothing to update
            #     db_state.append(tokenizer.encode("[db_state0]"))
            # else:
            #     turn_domain = 'general'
            #     for domain in constraint_dict_updates[bi].keys():
            #         #the last updated domain
            #         turn_domain=domain
            # follow damd for fair comparison
            db_vector = reader.bspan_to_DBpointer(tokenizer.decode(bspn_list), turn_domain[bi])
            if sum(db_vector) == 0:
                db_state.append(tokenizer.encode("[db_state0]"))
            else:
                db_state.append([tokenizer.encode("[db_state0]")[0] + db_vector.index(1) + 1])
            # use gold booking pointer, because we cannot issue BOOKING API

            if db[bi][0] >= tokenizer.encode("[db_state0+bookfail]")[0]:
                if db[bi][0] >= tokenizer.encode("[db_state0+booksuccess]")[0]:
                    db_state[-1][0] += 10
                else:
                    db_state[-1][0] += 5

        db_state = torch.tensor(
            db_state,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        resp_outputs = self.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     eos_token_id=tokenizer.encode("<eos_r>")[0],
                                     decoder_start_token_id=db_state,
                                     max_length=200,
                                     )

        resp_outputs = resp_outputs[:, 1:].tolist()  # skip DB state
        # print("DST:", tokenizer.decode(dst_outputs[0]))
        # print("RESP:", tokenizer.decode(resp_outputs[0]))
        return dst_outputs, resp_outputs  # , dst_time, length

    def inference_sequicity(
            self,
            tokenizer,
            reader,
            prev,
            input_ids=None,
            attention_mask=None,
            turn_domain=None,
            db=None
    ):
        # start = time.time()
        dst_outputs = self.generate(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    eos_token_id=tokenizer.encode("<eos_b>")[0],
                                    decoder_start_token_id=self.config.decoder_start_token_id,
                                    max_length=200,
                                    )
        # dst_time = time.time() - start
        # print(dst_time)
        dst_outputs = dst_outputs.tolist()
        # length = len(dst_outputs[0])
        # compute the DB state using the updated domain
        db_state = []
        for bi, bspn_list in enumerate(dst_outputs):
            db_vector = reader.bspan_to_DBpointer(tokenizer.decode(bspn_list), turn_domain[bi])
            if sum(db_vector) == 0:
                db_state.append(tokenizer.encode("[db_state0]"))
            else:
                db_state.append([tokenizer.encode("[db_state0]")[0] + db_vector.index(1) + 1])
            # use gold booking pointer, because we cannot issue BOOKING API

            if db[bi][0] >= tokenizer.encode("[db_state0+bookfail]")[0]:
                if db[bi][0] >= tokenizer.encode("[db_state0+booksuccess]")[0]:
                    db_state[-1][0] += 10
                else:
                    db_state[-1][0] += 5

        db_state = torch.tensor(
            db_state,
            dtype=torch.long,
            device=next(self.parameters()).device,
        )

        resp_outputs = self.generate(input_ids=input_ids,
                                     attention_mask=attention_mask,
                                     eos_token_id=tokenizer.encode("<eos_r>")[0],
                                     decoder_start_token_id=db_state,
                                     max_length=200,
                                     )

        resp_outputs = resp_outputs[:, 1:].tolist()  # skip DB state
        # print("DST:", tokenizer.decode(dst_outputs[0]))
        # print("RESP:", tokenizer.decode(resp_outputs[0]))
        return dst_outputs, resp_outputs  # , dst_time, length
