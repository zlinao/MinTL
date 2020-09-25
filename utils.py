
import os, random, csv, time, logging, json, re
from collections import Counter
import numpy as np
from itertools import chain
from copy import deepcopy
import spacy
from collections import OrderedDict
import torch

from damd_multiwoz import ontology
from damd_multiwoz.db_ops import MultiWozDB
from damd_multiwoz.config import global_config as cfg

class Vocab(object):
    def __init__(self, model, tokenizer):
        self.special_tokens = ["pricerange", "<pad>", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]", "<None>", "[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","[db_state0+bookfail]", "[db_state1+bookfail]","[db_state2+bookfail]","[db_state3+bookfail]","[db_state4+bookfail]", "[db_state0+booksuccess]","[db_state1+booksuccess]","[db_state2+booksuccess]","[db_state3+booksuccess]","[db_state4+booksuccess]"]
        self.attr_special_tokens = {'pad_token': '<pad>',
                         'additional_special_tokens': ["pricerange", "<go_r>", "<unk>", "<go_b>", "<go_a>", "<eos_u>", "<eos_r>", "<eos_b>", "<eos_a>", "<go_d>",
                    "[restaurant]","[hotel]","[attraction]","[train]","[taxi]","[police]","[hospital]","[general]","[inform]","[request]",
                    "[nooffer]","[recommend]","[select]","[offerbook]","[offerbooked]","[nobook]","[bye]","[greet]","[reqmore]","[welcome]",
                    "[value_name]","[value_choice]","[value_area]","[value_price]","[value_type]","[value_reference]","[value_phone]","[value_address]",
                    "[value_food]","[value_leave]","[value_postcode]","[value_id]","[value_arrive]","[value_stars]","[value_day]","[value_destination]",
                    "[value_car]","[value_departure]","[value_time]","[value_people]","[value_stay]","[value_pricerange]","[value_department]", "<None>", "[db_state0]","[db_state1]","[db_state2]","[db_state3]","[db_state4]","[db_state0+bookfail]", "[db_state1+bookfail]","[db_state2+bookfail]","[db_state3+bookfail]","[db_state4+bookfail]", "[db_state0+booksuccess]","[db_state1+booksuccess]","[db_state2+booksuccess]","[db_state3+booksuccess]","[db_state4+booksuccess]"]}
        self.tokenizer = tokenizer
        self.vocab_size = self.add_special_tokens_(model, tokenizer)


    def add_special_tokens_(self, model, tokenizer):
        """ Add special tokens to the tokenizer and the model if they have not already been added. """
        #orig_num_tokens = model.config.vocab_size  #some of experiments use this...
        orig_num_tokens = len(tokenizer)
        num_added_tokens = tokenizer.add_special_tokens(self.attr_special_tokens) # doesn't add if they are already there
        
        if num_added_tokens > 0:
            model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)
            model.tie_decoder()
        # print(orig_num_tokens)
        # print(num_added_tokens)

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
        return puntuation_handler(text)

# T5 cannot seperate the puntuation for some reason
def puntuation_handler(text):
    text = text.replace("'s", " 's")
    text = text.replace(".", " .")
    text = text.replace("!", " !")
    text = text.replace(",", " ,")
    text = text.replace("?", " ?")
    text = text.replace(":", " :")
    return text

class _ReaderBase(object):

    def __init__(self):
        self.train, self.dev, self.test = [], [], []
        self.vocab = None
        self.db = None

    def _bucket_by_turn(self, encoded_data):
        turn_bucket = {}
        for dial in encoded_data:
            turn_len = len(dial)
            if turn_len not in turn_bucket:
                turn_bucket[turn_len] = []
            turn_bucket[turn_len].append(dial)
        del_l = []
        for k in turn_bucket:
            if k >= 5: del_l.append(k)
            logging.debug("bucket %d instance %d" % (k, len(turn_bucket[k])))
        # for k in del_l:
        #    turn_bucket.pop(k)
        return OrderedDict(sorted(turn_bucket.items(), key=lambda i:i[0]))


    def _construct_mini_batch(self, data):
        all_batches = []
        batch = []
        for dial in data:
            batch.append(dial)
            #print(f"batch_size{cfg.batch_size}")
            if len(batch) == cfg.batch_size:
                # print('batch size: %d, batch num +1'%(len(batch)))
                all_batches.append(batch)
                batch = []
        # if remainder < 1/10 batch_size, just put them in the previous batch, otherwise form a new batch
        # print('last batch size: %d, batch num +1'%(len(batch)))
        if (len(batch)%len(cfg.cuda_device)) != 0:
            batch = batch[:-(len(batch)%len(cfg.cuda_device))]
        if len(batch) > 0.1 * cfg.batch_size:
            all_batches.append(batch)
        elif len(all_batches):
            all_batches[-1].extend(batch)
        else:
            all_batches.append(batch)
        return all_batches

    def transpose_batch(self, batch):
        dial_batch = []
        turn_num = len(batch[0])
        for turn in range(turn_num):
            turn_l = {}
            for dial in batch:
                this_turn = dial[turn]
                for k in this_turn:
                    if k not in turn_l:
                        turn_l[k] = []
                    turn_l[k].append(this_turn[k])
            dial_batch.append(turn_l)
        return dial_batch

    def inverse_transpose_batch(self, turn_batch_list):
        """
        :param turn_batch_list: list of transpose dial batch
        """
        dialogs = {}
        total_turn_num = len(turn_batch_list)
        # initialize
        for idx_in_batch, dial_id in enumerate(turn_batch_list[0]['dial_id']):
            dialogs[dial_id] = []
            for turn_n in range(total_turn_num):
                dial_turn = {}
                turn_batch = turn_batch_list[turn_n]
                for key, v_list in turn_batch.items():
                    if key == 'dial_id':
                        continue
                    value = v_list[idx_in_batch]
                    if key == 'pointer' and self.db is not None:
                        turn_domain = turn_batch['turn_domain'][idx_in_batch][-1]
                        value = self.db.pointerBack(value, turn_domain)
                    dial_turn[key] = value
                dialogs[dial_id].append(dial_turn)
        return dialogs


    def get_batches(self, set_name):
        global dia_count
        log_str = ''
        name_to_set = {'train': self.train, 'test': self.test, 'dev': self.dev}
        dial = name_to_set[set_name]
        turn_bucket = self._bucket_by_turn(dial)
        # self._shuffle_turn_bucket(turn_bucket)
        all_batches = []
        for k in turn_bucket:
            if set_name != 'test' and k==1 or k>=17:
                continue
            batches = self._construct_mini_batch(turn_bucket[k])
            log_str += "turn num:%d, dial num: %d, batch num: %d last batch len: %d\n"%(
                    k, len(turn_bucket[k]), len(batches), len(batches[-1]))
            # print("turn num:%d, dial num:v%d, batch num: %d, "%(k, len(turn_bucket[k]), len(batches)))
            all_batches += batches
        log_str += 'total batch num: %d\n'%len(all_batches)
        # print('total batch num: %d'%len(all_batches))
        # print('dialog count: %d'%dia_count)
        # return all_batches
        random.shuffle(all_batches)
        for i, batch in enumerate(all_batches):
            yield self.transpose_batch(batch)


    def save_result(self, write_mode, results, field, write_title=False):
        with open(cfg.result_path, write_mode) as rf:
            if write_title:
                rf.write(write_title+'\n')
            writer = csv.DictWriter(rf, fieldnames=field)
            writer.writeheader()
            writer.writerows(results)
        return None

    def save_result_report(self, results):

        ctr_save_path = cfg.result_path[:-4] + '_report_ctr%s.csv'%cfg.seed
        write_title = False if os.path.exists(ctr_save_path) else True
        if cfg.aspn_decode_mode == 'greedy':
            setting = ''
        elif cfg.aspn_decode_mode == 'beam':
            setting = 'width=%s'%str(cfg.beam_width)
            if cfg.beam_diverse_param>0:
                setting += ', penalty=%s'%str(cfg.beam_diverse_param)
        elif cfg.aspn_decode_mode == 'topk_sampling':
            setting = 'topk=%s'%str(cfg.topk_num)
        elif cfg.aspn_decode_mode == 'nucleur_sampling':
            setting = 'p=%s'%str(cfg.nucleur_p)
        res = {'exp': cfg.eval_load_path, 'true_bspn':cfg.use_true_curr_bspn, 'true_aspn': cfg.use_true_curr_aspn,
                  'decode': cfg.aspn_decode_mode, 'param':setting, 'nbest': cfg.nbest, 'selection_sheme': cfg.act_selection_scheme,
                  'match': results[0]['match'], 'success': results[0]['success'], 'bleu': results[0]['bleu'], 'act_f1': results[0]['act_f1'],
                  'avg_act_num': results[0]['avg_act_num'], 'avg_diverse': results[0]['avg_diverse_score']}
        with open(ctr_save_path, 'a') as rf:
            writer = csv.DictWriter(rf, fieldnames=list(res.keys()))
            if write_title:
                writer.writeheader()
            writer.writerows([res])

class MultiWozReader(_ReaderBase):
    def __init__(self, vocab=None, args=None):
        super().__init__()
        self.nlp = spacy.load('en_core_web_sm')
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
        data_fraction = self.args.fraction
        train_count = 0
   
        for fn, dial in self.data.items():
            if 'all' in cfg.exp_domains or self.exp_files.get(fn):
                if self.dev_files.get(fn):
                    self.dev.append(self._get_encoded_data(fn, dial))
                elif self.test_files.get(fn):
                    self.test.append(self._get_encoded_data(fn, dial))
                else:
                    if train_count>round(data_fraction*8438):
                        continue
                    self.train.append(self._get_encoded_data(fn, dial))
                    train_count+=1
    
        random.shuffle(self.train)
        random.shuffle(self.dev)
        random.shuffle(self.test)

    def _get_encoded_data(self, fn, dial):
        encoded_dial = []
        dial_context = []
        delete_op = self.vocab.tokenizer.encode("<None>") #delete operation
        prev_constraint_dict = {}
        for idx, t in enumerate(dial['log']):
            enc = {}
            enc['dial_id'] = fn
            #enc['user'] = self.vocab.tokenizer.encode(t['user']) + self.vocab.tokenizer.encode(['<eos_u>'])
            dial_context.append( self.vocab.tokenizer.encode(t['user']) + self.vocab.tokenizer.encode('<eos_u>') )
            enc['user'] = list(chain(*dial_context[-self.args.context_window:])) # here we use user to represent dialogue history
            enc['usdx'] = self.vocab.tokenizer.encode(t['user_delex']) + self.vocab.tokenizer.encode('<eos_u>')
            enc['resp'] = self.vocab.tokenizer.encode(t['resp']) + self.vocab.tokenizer.encode('<eos_r>')
            enc['resp_nodelex'] = self.vocab.tokenizer.encode(t['resp_nodelex']) + self.vocab.tokenizer.encode('<eos_r>')
            enc['bspn'] = self.vocab.tokenizer.encode(t['constraint']) + self.vocab.tokenizer.encode('<eos_b>')
            constraint_dict = self.bspan_to_constraint_dict(t['constraint'])
            update_bspn = self.check_update(prev_constraint_dict, constraint_dict)
            enc['update_bspn'] = self.vocab.tokenizer.encode(update_bspn)
            #'bspn': '[hotel] area north type guest house stay 5 day tuesday people 5 [train] leave sunday destination london liverpool street departure cambridge', 
            enc['bsdx'] = self.vocab.tokenizer.encode(t['cons_delex']) + self.vocab.tokenizer.encode('<eos_b>')
            enc['aspn'] = self.vocab.tokenizer.encode(t['sys_act']) + self.vocab.tokenizer.encode('<eos_a>')
            enc['dspn'] = self.vocab.tokenizer.encode(t['turn_domain']) + self.vocab.tokenizer.encode('<eos_d>')
            enc['pointer'] = [int(i) for i in t['pointer'].split(',')]
            # print(self.vocab.tokenizer.encode("[db_state0]"))
            # print(self.vocab.tokenizer.encode("[db_state4]"))
            if sum(enc['pointer'][:-2])==0:
                enc['input_pointer'] = self.vocab.tokenizer.encode("[db_state0]")
            else:
                enc['input_pointer'] = [self.vocab.tokenizer.encode("[db_state0]")[0] + enc['pointer'][:-2].index(1)+1] 
            if sum(enc['pointer'][-2:])>0:
                enc['input_pointer'][0] += (enc['pointer'][-2:].index(1)+1) * 5 # 5 means index(db_state0+bookfail)-index(db_state0)=5

                
            enc['turn_domain'] = t['turn_domain'].split()
            enc['turn_num'] = t['turn_num']
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

    def bspan_to_DBpointer(self, bspan, turn_domain):
        constraint_dict = self.bspan_to_constraint_dict(bspan)
        # follow damd
        matnums = self.db.get_match_num(constraint_dict)
        match_dom = turn_domain[0] if len(turn_domain) == 1 else turn_domain[1]
        match_dom = match_dom[1:-1] if match_dom.startswith('[') else match_dom
        match = matnums[match_dom]
        vector = self.db.addDBPointer(match_dom, match)
        return vector


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
        DB state: ['input_pointer']
        output1: dialogue state update ['update_bspn'] or current dialogue state ['bspn']
        output2: dialogue response ['resp']
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
        response, response_input = self.padOutput(batch['resp'], pad_token)
        inputs["state_update"] = torch.tensor(state_update,dtype=torch.long) # batch_size, seq_len
        inputs["response"] = torch.tensor(response,dtype=torch.long)
        inputs["state_input"] = torch.tensor(np.concatenate( (np.ones((batch_size,1))*dst_start_token  , state_input[:,:-1]), axis=1 ) ,dtype=torch.long)
        inputs["response_input"] = torch.tensor( np.concatenate( ( np.array(batch['input_pointer']), response_input[:,:-1]), axis=1 ) ,dtype=torch.long)
        inputs["turn_domain"] = batch["turn_domain"]
        inputs["input_pointer"] = torch.tensor(np.array(batch['input_pointer']),dtype=torch.long)
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

        if cfg.bspn_mode == 'bspn':
            field = ['dial_id', 'turn_num', 'user', 'bspn_gen','bspn', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'pointer']
        elif not cfg.enable_dst:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen','bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'bspn', 'pointer']
        else:
            field = ['dial_id', 'turn_num', 'user', 'bsdx_gen','bsdx', 'resp_gen', 'resp', 'aspn_gen', 'aspn',
                        'dspn_gen', 'dspn', 'bspn_gen','bspn', 'pointer']
        # if self.multi_acts_record is not None:
        #     field.insert(7, 'multi_act_gen')

        for dial_id, turns in result_dict.items():
            entry = {'dial_id': dial_id, 'turn_num': len(turns)}
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

    # def restore(self, resp, domain, constraint_dict, mat_ents):
    #     restored = resp

    #     restored = restored.replace('[value_reference]', '53022')
    #     restored = restored.replace('[value_car]', 'BMW')

    #     # restored.replace('[value_phone]', '830-430-6666')
    #     for d in domain:
    #         constraint = constraint_dict.get(d,None)
    #         if constraint:
    #             if 'stay' in constraint:
    #                 restored = restored.replace('[value_stay]', constraint['stay'])
    #             if 'day' in constraint:
    #                 restored = restored.replace('[value_day]', constraint['day'])
    #             if 'people' in constraint:
    #                 restored = restored.replace('[value_people]', constraint['people'])
    #             if 'time' in constraint:
    #                 restored = restored.replace('[value_time]', constraint['time'])
    #             if 'type' in constraint:
    #                 restored = restored.replace('[value_type]', constraint['type'])
    #             if d in mat_ents and len(mat_ents[d])==0:
    #                 for s in constraint:
    #                     if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
    #                         restored = restored.replace('[value_price]', constraint['pricerange'])
    #                     if s+']' in restored:
    #                         restored = restored.replace('[value_%s]'%s, constraint[s])

    #         if '[value_choice' in restored and mat_ents.get(d):
    #             restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
    #     if '[value_choice' in restored:
    #         restored = restored.replace('[value_choice]', '3')


    #     # restored.replace('[value_car]', 'BMW')


    #     try:
    #         ent = mat_ents.get(domain[-1], [])
    #         if ent:
    #             ent = ent[0]

    #             for t in restored.split():
    #                 if '[value' in t:
    #                     slot = t[7:-1]
    #                     if ent.get(slot):
    #                         if domain[-1] == 'hotel' and slot == 'price':
    #                             slot = 'pricerange'
    #                         restored = restored.replace(t, ent[slot])
    #                     elif slot == 'price':
    #                         if ent.get('pricerange'):
    #                             restored = restored.replace(t, ent['pricerange'])
    #                         else:
    #                             print(restored, domain)
    #     except:
    #         print(resp)
    #         print(restored)
    #         quit()


    #     restored = restored.replace('[value_phone]', '62781111')
    #     restored = restored.replace('[value_postcode]', 'CG9566')
    #     restored = restored.replace('[value_address]', 'Parkside, Cambridge')

    #     return restored

    def restore(self, resp, domain, constraint_dict):
        restored = resp
        restored = restored.capitalize()
        restored = restored.replace(' -s', 's')
        restored = restored.replace(' -ly', 'ly')
        restored = restored.replace(' -er', 'er')

        mat_ents = self.db.get_match_num(constraint_dict, True)
        self.delex_refs = ["w29zp27k","qjtixk8c","wbjgaot8","wjxw4vrv","sa63gzjd","i4afi8et","u595dz8a","8ttxct27","vcmkko1k","a5litxvz","2gy5ulll","gethuntl","i76goxin","mq7amf1m","isyr3hnc","69srbpnj","pmhz3tjo","5vrjsmse","ie05gdqs","wpa3iy8c","lnk1guuk","bbg39tvv","73mseuiq","6knjsqxy","znl8d0eg","4rz5lydp","r9xjc41b","d77jcgj2","sw8ac8gh",]
        ref =  random.choice(self.delex_refs)
        restored = restored.replace('[value_reference]', ref.upper())
        restored = restored.replace('[value_car]', 'BMW')

        # restored.replace('[value_phone]', '830-430-6666')
        for d in domain:
            constraint = constraint_dict.get(d,None)
            if constraint:
                if 'stay' in constraint:
                    restored = restored.replace('[value_stay]', constraint['stay'])
                if 'day' in constraint:
                    restored = restored.replace('[value_day]', constraint['day'])
                if 'people' in constraint:
                    restored = restored.replace('[value_people]', constraint['people'])
                if 'time' in constraint:
                    restored = restored.replace('[value_time]', constraint['time'])
                if 'type' in constraint:
                    restored = restored.replace('[value_type]', constraint['type'])
                if d in mat_ents and len(mat_ents[d])==0:
                    for s in constraint:
                        if s == 'pricerange' and d in ['hotel', 'restaurant'] and 'price]' in restored:
                            restored = restored.replace('[value_price]', constraint['pricerange'])
                        if s+']' in restored:
                            restored = restored.replace('[value_%s]'%s, constraint[s])

            if '[value_choice' in restored and mat_ents.get(d):
                restored = restored.replace('[value_choice]', str(len(mat_ents[d])))
        if '[value_choice' in restored:
            restored = restored.replace('[value_choice]', str(random.choice([1,2,3,4,5])))


        # restored.replace('[value_car]', 'BMW')
        stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]

        ent = mat_ents.get(domain[-1], [])
        if ent:
            # handle multiple [value_xxx] tokens first
            restored_split = restored.split()
            token_count = Counter(restored_split)
            for idx, t in enumerate(restored_split):
                if '[value' in t and token_count[t]>1 and token_count[t]<=len(ent):
                    slot = t[7:-1]
                    pattern = r'\['+t[1:-1]+r'\]'
                    for e in ent:
                        if e.get(slot):
                            if domain[-1] == 'hotel' and slot == 'price':
                                slot = 'pricerange'
                            if slot in ['name', 'address']:
                                rep = ' '.join([i.capitalize() if i not in stopwords else i for i in e[slot].split()])
                            elif slot in ['id','postcode']:
                                rep = e[slot].upper()
                            else:
                                rep = e[slot]
                            restored = re.sub(pattern, rep, restored, 1)
                        elif slot == 'price' and  e.get('pricerange'):
                            restored = re.sub(pattern, e['pricerange'], restored, 1)

            # handle normal 1 entity case
            ent = ent[0]
            for t in restored.split():
                if '[value' in t:
                    slot = t[7:-1]
                    if ent.get(slot):
                        if domain[-1] == 'hotel' and slot == 'price':
                            slot = 'pricerange'
                        if slot in ['name', 'address']:
                            rep = ' '.join([i.capitalize() if i not in stopwords else i for i in ent[slot].split()])
                        elif slot in ['id','postcode']:
                            rep = ent[slot].upper()
                        else:
                            rep = ent[slot]
                        # rep = ent[slot]
                        restored = restored.replace(t, rep)
                        # restored = restored.replace(t, ent[slot])
                    elif slot == 'price' and  ent.get('pricerange'):
                        restored = restored.replace(t, ent['pricerange'])
                        # else:
                        #     print(restored, domain)
        restored = restored.replace('[value_phone]', '07338019809')#taxi number need to get from api call, which is not available
        for t in restored.split():
            if '[value' in t:
                restored = restored.replace(t, 'UNKNOWN')

        restored = restored.split()
        for idx, w in enumerate(restored):
            if idx>0 and restored[idx-1] in ['.', '?', '!']:
                restored[idx]= restored[idx].capitalize()
        restored = ' '.join(restored)
        return restored


    def relex(self, result_path, output_path):
        data = []

        with open(result_path, "r") as f:
            reader = csv.reader(f, delimiter=',')
            for i, row in enumerate(reader):
                if i == 10: # skip statistic ressults
                    namelist = row
                elif i > 10:
                    data.append(row)

        bspn_index = namelist.index("bspn_gen")
        resp_index = namelist.index("resp_gen")
        dspn_index = namelist.index("dspn_gen")

        row_list = []
        row_list.append(namelist)

        for row in data:
            bspn = row[bspn_index]
            resp = row[resp_index]
            dspn = [row[dspn_index].replace("[","").replace("]","")]
            if bspn == "" or resp == "":
                row_list.append(row)
            else:
                constraint_dict = self.bspan_to_constraint_dict(bspn)
                new_resp_gen = self.restore(resp, dspn, constraint_dict)

                row[resp_index] = new_resp_gen
                row_list.append(row)

                
                print("resp", resp)
                #print("cons_dict: ", cons_dict)
                #print("dspn: ", dspn)
                print("new_resp_gen: ", new_resp_gen)

        with open(output_path, "w") as fw:
            writer = csv.writer(fw)
            writer.writerows(row_list)
    
