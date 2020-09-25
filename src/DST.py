import os, random, argparse, time, logging, json, tqdm
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch
from itertools import chain
from copy import deepcopy


from utils import _ReaderBase
from damd_multiwoz import ontology
from damd_multiwoz.db_ops import MultiWozDB
from damd_multiwoz.config import global_config as cfg
from damd_multiwoz.eval import MultiWozEvaluator
from transformers import (AdamW, T5Tokenizer, T5ForConditionalGeneration, BartTokenizer, BartForConditionalGeneration, WEIGHTS_NAME,CONFIG_NAME, get_linear_schedule_with_warmup)

class BartTokenizer(BartTokenizer):
    def encode(self,text,add_special_tokens=False):
        encoded_inputs = self.encode_plus(text,add_special_tokens=False)
        return encoded_inputs["input_ids"]
class BART_DST(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def inference(
        self,
        tokenizer,
        reader,
        prev,
        input_ids=None,
        attention_mask=None,
        turn_domain=None,
    ):  

        dst_outputs = self.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            eos_token_id=tokenizer.encode("<eos_b>")[0],
                            decoder_start_token_id=self.config.decoder_start_token_id,
                            max_length=200,
                            min_length=1,
                            num_beams=1,
                            length_penalty=1.0,
                            )
        dst_outputs = dst_outputs.tolist()
        # DST_UPDATE -> DST
        #check whether need to add eos
        #dst_outputs = [dst+tokenizer.encode("<eos_b>") for dst in dst_outputs]
        batch_size = input_ids.shape[0]
        constraint_dict_updates = [reader.bspan_to_constraint_dict(tokenizer.decode(dst_outputs[i])) for i in range(batch_size)]

        if prev['bspn']:
            # update the belief state
            dst_outputs = [reader.update_bspn(prev_bspn=prev['bspn'][i], bspn_update=dst_outputs[i]) for i in range(batch_size)]
        
        return dst_outputs 


class T5_DST(T5ForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
    
    def inference(
        self,
        tokenizer,
        reader,
        prev,
        input_ids=None,
        attention_mask=None,
        turn_domain=None,
    ):  

        dst_outputs = self.generate(input_ids=input_ids,
                            attention_mask=attention_mask,
                            eos_token_id=tokenizer.encode("<eos_b>")[0],
                            decoder_start_token_id=self.config.decoder_start_token_id,
                            max_length=200,
                            )
        dst_outputs = dst_outputs.tolist()
        # DST_UPDATE -> DST
        #check whether need to add eos
        #dst_outputs = [dst+tokenizer.encode("<eos_b>") for dst in dst_outputs]
        batch_size = input_ids.shape[0]
        constraint_dict_updates = [reader.bspan_to_constraint_dict(tokenizer.decode(dst_outputs[i])) for i in range(batch_size)]

        if prev['bspn']:
            # update the belief state
            dst_outputs = [reader.update_bspn(prev_bspn=prev['bspn'][i], bspn_update=dst_outputs[i]) for i in range(batch_size)]
        
        return dst_outputs 

class Vocab(object):
    def __init__(self, model, tokenizer):
        self.special_tokens = ["pricerange", "<pad>", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]", "[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","<None>"]
        self.attr_special_tokens = {'pad_token': '<pad>',
                         'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]","[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","<None>"]}
        self.tokenizer = tokenizer
        self.vocab_size = self.add_special_tokens_(model, tokenizer)


    def add_special_tokens_(self, model, tokenizer):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        #orig_num_tokens = model.config.vocab_size
        orig_num_tokens = len(tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(self.attr_special_tokens) # doesn't add if they are already there
        if num_added_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
        return orig_num_tokens + num_added_tokens

    def encode(self, word):
        """ customize for damd script """
        return self.tokenizer.encode(word)[0]

    def sentence_encode(self, word_list):
        """ customize for damd script """
        return self.tokenizer.encode(" ".join(word_list))

    def decode(self, idx):
        """ customize for damd script """
        return self.tokenizer.decode(idx)

    def sentence_decode(self, index_list, eos=None):
        """ customize for damd script """
        l = self.tokenizer.decode(index_list)
        l = l.split()
        if not eos or eos not in l:
            text = ' '.join(l)
        else:
            idx = l.index(eos)
            text = ' '.join(l[:idx])
        return text

class MultiWozReader(_ReaderBase):
    def __init__(self, vocab=None, args=None):
        super().__init__()
        self.db = MultiWozDB(cfg.dbs)
        self.args = args
        self.domain_files = json.loads(open(cfg.domain_file_path, 'r').read())
        self.slot_value_set = json.loads(open(cfg.slot_value_set_path, 'r').read())
        test_list = [l.strip().lower() for l in open(cfg.test_list, 'r').readlines()]
        dev_list = [l.strip().lower() for l in open(cfg.dev_list, 'r').readlines()]
        self.dev_files, self.test_files = {}, {}
        for fn in test_list:
            self.test_files[fn.replace('.json', '')] = 1
        for fn in dev_list:
            self.dev_files[fn.replace('.json', '')] = 1

        self.vocab = vocab
        self.vocab_size = vocab.vocab_size

        self._load_data()


    def _load_data(self, save_temp=False):
        self.data = json.loads(open(cfg.data_path+cfg.data_file, 'r', encoding='utf-8').read().lower())
        self.train, self.dev, self.test = [] , [], []
        for fn, dial in self.data.items():
            if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                if self.dev_files.get(fn):
                    self.dev.append(self._get_encoded_data(fn, dial))
                elif self.test_files.get(fn):
                    self.test.append(self._get_encoded_data(fn, dial))
                else:
                    self.train.append(self._get_encoded_data(fn, dial))

        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        dial_context = []
        delete_op = self.vocab.tokenizer.encode("<None>") #[32157]
        prev_constraint_dict = {}
        for idx, t in enumerate(dial['log']):
            enc = {}
            enc['dial_id'] = fn
            dial_context.append( self.vocab.tokenizer.encode(t['user']) + self.vocab.tokenizer.encode('<eos_u>') )
            enc['resp_nodelex'] = self.vocab.tokenizer.encode(t['resp_nodelex']) + self.vocab.tokenizer.encode('<eos_r>')
            enc['user'] = list(chain(*dial_context[-self.args.context_window:])) # here we use user to represent dialogue history
            enc['bspn'] = self.vocab.tokenizer.encode(t['constraint']) + self.vocab.tokenizer.encode('<eos_b>')
            constraint_dict = self.bspan_to_constraint_dict(t['constraint'])
            update_bspn = self.check_update(prev_constraint_dict, constraint_dict)
            enc['update_bspn'] = self.vocab.tokenizer.encode(update_bspn)            
            encoded_dial.append(enc)

            prev_constraint_dict = constraint_dict
            dial_context.append( enc['resp_nodelex'] )
        return encoded_dial

    def check_update(self, prev_constraint_dict, constraint_dict):
        update_dict = {}
        if prev_constraint_dict==constraint_dict:
            return '<eos_b>'
        for domain in constraint_dict:
            if domain in prev_constraint_dict:
                for slot in constraint_dict[domain]:
                    if constraint_dict[domain].get(slot) != prev_constraint_dict[domain].get(slot):
                        if domain not in update_dict:
                            update_dict[domain] = {}
                        update_dict[domain][slot] = constraint_dict[domain].get(slot)
                # if delete is needed
                # if len(prev_constraint_dict[domain])>len(constraint_dict[domain]):
                for slot in prev_constraint_dict[domain]:
                    if constraint_dict[domain].get(slot) is None:
                        update_dict[domain][slot] = "<None>"
            else:
                update_dict[domain] = deepcopy(constraint_dict[domain])
    

        update_bspn= self.constraint_dict_to_bspan(update_dict)
        return update_bspn

    def constraint_dict_to_bspan(self, constraint_dict):
        if not constraint_dict:
            return "<eos_b>"
        update_bspn=""
        for domain in constraint_dict:
            if len(update_bspn)==0: 
                update_bspn += f"[{domain}]"
            else:
                update_bspn += f" [{domain}]"
            for slot in constraint_dict[domain]:
                update_bspn += f" {slot} {constraint_dict[domain][slot]}"
        update_bspn += f" <eos_b>"
        return update_bspn

    def bspan_to_constraint_dict(self, bspan, bspn_mode = 'bspn'):
        # add decoded(str) here
        bspan = bspan.split() if isinstance(bspan, str) else bspan
        constraint_dict = {}
        domain = None
        conslen = len(bspan)
        for idx, cons in enumerate(bspan):
            cons = self.vocab.decode(cons) if type(cons) is not str else cons
            if cons == "[slot]":
                continue
            if cons == '<eos_b>':
                break
            if '[' in cons:
                if cons[1:-1] not in ontology.all_domains:
                    continue
                domain = cons[1:-1]
            elif cons in ontology.get_slot:
                if domain is None:
                    continue
                if cons == 'people':
                    # handle confusion of value name "people's portraits..." and slot people
                    try:
                        ns = bspan[idx+1]
                        ns = self.vocab.decode(ns) if type(ns) is not str else ns
                        if ns == "'s":
                            continue
                    except:
                        continue
                if not constraint_dict.get(domain):
                    constraint_dict[domain] = {}
                if bspn_mode == 'bsdx':
                    constraint_dict[domain][cons] = 1
                    continue
                vidx = idx+1
                if vidx == conslen:
                    break
                vt_collect = []
                vt = bspan[vidx]
                vt = self.vocab.decode(vt) if type(vt) is not str else vt
                while vidx < conslen and vt != '<eos_b>' and '[' not in vt and vt not in ontology.get_slot:
                    vt_collect.append(vt)
                    vidx += 1
                    if vidx == conslen:
                        break
                    vt = bspan[vidx]
                    vt = self.vocab.decode(vt) if type(vt) is not str else vt
                if vt_collect:
                    constraint_dict[domain][cons] = ' '.join(vt_collect)

        return constraint_dict


    def dspan_to_domain(self, dspan):
        domains = {}
        dspan = dspan.split() if isinstance(dspan, str) else dspan
        for d in dspan:
            dom = self.vocab.decode(d) if type(d) is not str else d
            if dom != '<eos_d>':
                domains[dom] = 1
            else:
                break
        return domains

    def convert_batch(self, batch, prev, first_turn=False, dst_start_token = 0):
        """
        user: dialogue history ['user']
        input: previous dialogue state + dialogue history
        output1: dialogue state update ['update_bspn'] or current dialogue state ['bspn']
        """
        inputs = {}
        pad_token = self.vocab.tokenizer.encode("<pad>")[0]
        batch_size = len(batch['user'])
        # input: previous dialogue state + dialogue history
        input_ids = []
        if first_turn:
            for i in range(batch_size):
                input_ids.append(self.vocab.tokenizer.encode('<eos_b>') + batch['user'][i])
        else:
            for i in range(batch_size):
                input_ids.append(prev['bspn'][i] + batch['user'][i])
        input_ids, masks = self.padInput(input_ids, pad_token)
        inputs["input_ids"] = torch.tensor(input_ids,dtype=torch.long)
        inputs["masks"] = torch.tensor(masks,dtype=torch.long)
        if self.args.noupdate_dst:
            # here we use state_update denote the belief span (bspn)...
            state_update, state_input = self.padOutput(batch['bspn'], pad_token)
        else:
            state_update, state_input = self.padOutput(batch['update_bspn'], pad_token)
        inputs["state_update"] = torch.tensor(state_update,dtype=torch.long) # batch_size, seq_len
        inputs["state_input"] = torch.tensor(np.concatenate( (np.ones((batch_size,1))*dst_start_token  , state_input[:,:-1]), axis=1 ) ,dtype=torch.long)
        # for k in inputs:
        #     if k=="masks":
        #         print(k)
        #         print(inputs[k])
        #     else:
        #         print(k)
        #         print(inputs[k].tolist())
        #         print(k)
        #         print(self.vocab.tokenizer.decode(inputs[k].tolist()[0]))
        
        return inputs

    def padOutput(self, sequences, pad_token):
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        max_len = max(lengths)
        output_ids = np.ones((num_samples, max_len)) * (-100) #-100 ignore by cross entropy
        decoder_inputs = np.ones((num_samples, max_len)) * pad_token
        for idx, s in enumerate(sequences):
            trunc = s[:max_len]
            output_ids[idx, :lengths[idx]] = trunc
            decoder_inputs[idx, :lengths[idx]] = trunc
        return output_ids, decoder_inputs

    def padInput(self, sequences, pad_token):
        lengths = [len(s) for s in sequences]
        num_samples = len(lengths)
        max_len = max(lengths)
        input_ids = np.ones((num_samples, max_len)) * pad_token
        masks = np.zeros((num_samples, max_len))

        for idx, s in enumerate(sequences):
            trunc = s[-max_len:]
            input_ids[idx, :lengths[idx]] = trunc
            masks[idx, :lengths[idx]] = 1
        return input_ids, masks

    def update_bspn(self, prev_bspn, bspn_update):
        constraint_dict_update = self.bspan_to_constraint_dict(self.vocab.tokenizer.decode(bspn_update) )
        if not constraint_dict_update:
            return prev_bspn
        constraint_dict = self.bspan_to_constraint_dict(self.vocab.tokenizer.decode(prev_bspn) )
        
        for domain in constraint_dict_update:
            if domain not in constraint_dict:
                constraint_dict[domain] = {}
            for slot, value in constraint_dict_update[domain].items():
                if value=="<None>": #delete the slot
                    _ = constraint_dict[domain].pop(slot, None)
                else:
                    constraint_dict[domain][slot]=value
        updated_bspn = self.vocab.tokenizer.encode(self.constraint_dict_to_bspan(constraint_dict))
        return updated_bspn

    def wrap_result(self, result_dict, eos_syntax=None):
        decode_fn = self.vocab.sentence_decode
        results = []
        eos_syntax = ontology.eos_tokens if not eos_syntax else eos_syntax

        field = ['dial_id', 'turn_num', 'user', 'bspn_gen','bspn']


        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
            # customize for the eval, always skip the first turn, so we create a dummy
            for prop in field[2:]:
                entry[prop] = ''
            results.append(entry)
            for turn_no, turn in enumerate(turns):
                entry = {'dial_id': dial_id}
                for key in field:
                    if key in ['dial_id']:
                        continue
                    v = turn.get(key, '')
                    if key == 'turn_domain':
                        v = ' '.join(v)
                    entry[key] = decode_fn(v, eos=eos_syntax[key]) if key in eos_syntax and v != '' else v
                results.append(entry)
        return results, field


class Model(object):
    def __init__(self, args, test=False):
        if args.back_bone=="t5":  
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = T5_DST.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        elif args.back_bone=="bart":
            self.tokenizer = BartTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = BART_DST.from_pretrained(args.model_path if test else args.pretrained_checkpoint)

        vocab = Vocab(self.model, self.tokenizer)
        self.reader = MultiWozReader(vocab,args)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args
        self.model.to(args.device)
        

    def load_model(self):
        # model_state_dict = torch.load(checkpoint)
        # start_model.load_state_dict(model_state_dict)
        if self.args.back_bone=="t5":
            self.model = T5_DST.from_pretrained(self.args.model_path)
        elif self.args.back_bone=="bart":
            self.model = BART_DST.from_pretrained(self.args.model_path)
        self.model.to(self.args.device)

    def train(self):
        btm = time.time()
        step = 0
        prev_min_loss = 1000
        print(f"vocab_size:{self.model.config.vocab_size}")
        torch.save(self.args, self.args.model_path + '/model_training_args.bin')
        self.tokenizer.save_pretrained(self.args.model_path)
        self.model.config.to_json_file(os.path.join(self.args.model_path, CONFIG_NAME))
        self.model.train()
        # lr scheduler
        lr_lambda = lambda epoch: self.args.lr_decay ** epoch
        scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)

        for epoch in range(cfg.epoch_num):
            log_dst = 0
            log_cnt = 0
            sw = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                py_prev = {'pv_bspn': None}
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                    for k in inputs:
                        
                        inputs[k] = inputs[k].to(self.args.device)

                    outputs = self.model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["state_input"],
                                        lm_labels=inputs["state_update"]
                                        )
                    dst_loss = outputs[0]

                    py_prev['bspn'] = turn_batch['bspn']

                    total_loss = dst_loss / self.args.gradient_accumulation_steps

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step+=1
                    log_dst +=float(dst_loss.item())
                    log_cnt += 1

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [bspn] loss: {:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                                        log_dst/(log_cnt+ 1e-8),
                                                                                        time.time()-btm,
                                                                                        turn_num+1))
            epoch_sup_loss = log_dst/(log_cnt+ 1e-8)
            do_test = False
            valid_loss = self.validate(do_test=do_test)
            logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))

            if valid_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                prev_min_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, WEIGHTS_NAME))
                logging.info('Model saved')
                #self.save_model(epoch)
            else:
                early_stop_count -= 1
                scheduler.step()
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))


                if not early_stop_count:
                    self.load_model()
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(self.args.model_path, 'eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    logging.info(str(cfg))
                    self.eval()
                    return

        self.load_model()
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(self.args.model_path, 'eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        logging.info(str(cfg))
        self.eval()


    def validate(self, data='dev', do_test=False):
        self.model.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data)
        result_collection = {}
        for batch_num, dial_batch in enumerate(data_iterator):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                for k in inputs:
                    inputs[k] = inputs[k].to(self.args.device)
                dst_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"])
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, _ = self.reader.wrap_result(result_collection)
        # print(results)
        jg, slot_f1, slot_acc, slot_cnt, slot_corr  = self.evaluator.dialog_state_tracking_eval(results, bspn_mode='bspn')
        logging.info('validation DST join goal: %2.1f  slot_f1: %2.1f  slot_acc: %2.1f'%(jg, slot_f1, slot_acc))
        self.model.train()
        if do_test:
            print('result preview...')
            self.eval()
        return 100-jg

    def eval(self, data='test'):
        self.model.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data)
        for batch_num, dial_batch in tqdm.tqdm(enumerate(data_iterator)):
            py_prev = {'bspn': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                for k in inputs:
                    inputs[k] = inputs[k].to(self.args.device)
                dst_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"])
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, field = self.reader.wrap_result(result_collection)
        jg, slot_f1, slot_acc, slot_cnt, slot_corr  = self.evaluator.dialog_state_tracking_eval(results, bspn_mode='bspn')
        logging.info('test DST join goal: %2.1f  slot_f1: %2.1f  slot_acc: %2.1f'%(jg, slot_f1, slot_acc))
        self.args.model_path
        with open(os.path.join(self.args.model_path, 'result.txt'), 'w') as f:
            f.write('test DST join goal: %2.1f  slot_f1: %2.1f  slot_acc: %2.1f'%(jg, slot_f1, slot_acc))
        # self.reader.metric_record(metric_results)
        self.model.train()
        return None


    def count_params(self):
        module_parameters = filter(lambda p: p.requires_grad, self.m.parameters())
        param_cnt = int(sum([np.prod(p.size()) for p in module_parameters]))

        print('total trainable params: %d' % param_cnt)
        return param_cnt


def parse_arg_cfg(args):
    if args.cfg:
        for pair in args.cfg:
            k, v = tuple(pair.split('='))
            dtype = type(getattr(cfg, k))
            if dtype == type(None):
                raise ValueError()
            if dtype is bool:
                v = False if v == 'False' else True
            elif dtype is list:
                v = v.split(',')
                if k=='cuda_device':
                    v = [int(no) for no in v]
            else:
                v = dtype(v)
            setattr(cfg, k, v)
    return


def main():
    if not os.path.exists('./experiments_DST'):
        os.mkdir('./experiments_DST')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default="t5-small", help="Path, url or short name of the model")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--context_window", type=int, default=5, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--back_bone", type=str, default="t5", help="choose t5 or bart") 
    parser.add_argument("--noupdate_dst", action='store_true', help="dont use update base DST")
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test':
        parse_arg_cfg(args)
        cfg_load = json.loads(open(os.path.join(args.model_path, 'exp_cfg.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_per_domain', 'use_true_pv_resp',
                        'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                        'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                        'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                        'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                        'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                        'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                        'act_selection_scheme', 'beam_penalty_type', 'record_mode']:
                continue
            setattr(cfg, k, v)
            cfg.result_path = os.path.join(args.model_path, 'result.csv')
    else:
        parse_arg_cfg(args)
        if args.model_path=="":
            args.model_path = 'experiments_DST/{}_sd{}_lr{}_bs{}_sp{}_dc{}_cw{}_model_{}_noupdate{}/'.format('-'.join(cfg.exp_domains), cfg.seed, args.lr, cfg.batch_size,
                                                                                            cfg.early_stop_count, args.lr_decay, args.context_window, args.pretrained_checkpoint, args.noupdate_dst)
        if not os.path.exists(args.model_path):
            os.makedirs(args.model_path)
        cfg.result_path = os.path.join(args.model_path, 'result.csv')
        cfg.eval_load_path = args.model_path

    cfg._init_logging_handler(args.mode)

    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    #cfg.model_parameters = m.count_params()
    logging.info(str(cfg))

    if args.mode == 'train':
        with open(os.path.join(args.model_path, 'exp_cfg.json'), 'w') as f:
            json.dump(cfg.__dict__, f, indent=2)
        m = Model(args)
        m.train()
    elif args.mode == 'test':
        m = Model(args,test=True)
        m.eval(data='test')



if __name__ == '__main__':
    main()
