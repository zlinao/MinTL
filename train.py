import os, random, argparse, time, logging, json, tqdm
import numpy as np
from copy import deepcopy
from collections import OrderedDict
import torch

from utils import Vocab, MultiWozReader
from damd_multiwoz.config import global_config as cfg
from damd_multiwoz.eval import MultiWozEvaluator
from transformers import (AdamW, T5Tokenizer, BartTokenizer, WEIGHTS_NAME,CONFIG_NAME, get_linear_schedule_with_warmup)
from T5 import MiniT5
from BART import MiniBART

class BartTokenizer(BartTokenizer):
    def encode(self,text,add_special_tokens=False):
        encoded_inputs = self.encode_plus(text,add_special_tokens=False)
        return encoded_inputs["input_ids"]



class Model(object):
    def __init__(self, args, test=False):
        if args.back_bone=="t5":  
            self.tokenizer = T5Tokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = MiniT5.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        elif args.back_bone=="bart":
            self.tokenizer = BartTokenizer.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
            self.model = MiniBART.from_pretrained(args.model_path if test else args.pretrained_checkpoint)
        vocab = Vocab(self.model, self.tokenizer)
        self.reader = MultiWozReader(vocab,args)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        self.optim = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args
        self.model.to(args.device)
        

    def load_model(self):
        # model_state_dict = torch.load(checkpoint)
        # start_model.load_state_dict(model_state_dict)
        self.model = type(self.model).from_pretrained(self.args.model_path)
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
            log_loss = 0
            log_dst = 0
            log_resp = 0
            log_cnt = 0
            sw = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                py_prev = {'pv_bspn': None}
                for turn_num, turn_batch in enumerate(dial_batch):
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn, dst_start_token=self.model.config.decoder_start_token_id)
                    for k in inputs:
                        if k!="turn_domain":
                            inputs[k] = inputs[k].to(self.args.device)

                    #     print(k)
                    #     print(inputs[k])
 
                    # batch_size = inputs["input_ids"].shape[0]
                    # input_seq_len = inputs["input_ids"].shape[1]
                    # dst_seq_len = inputs["state_input"].shape[1]
                    # resp_seq_len = inputs["response"].shape[1]
                    # print(f"batch_size:{batch_size},seq_len:{input_seq_len}, dst:{dst_seq_len}, resp:{resp_seq_len}")
                    outputs = self.model(input_ids=inputs["input_ids"],
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["state_input"],
                                        lm_labels=inputs["state_update"]
                                        )
                    dst_loss = outputs[0]

            
                    # outputs = self.model(input_ids=inputs["input_ids"],
                    #                     attention_mask=inputs["masks"],
                    #                     decoder_input_ids=inputs["response_input"],
                    #                     lm_labels=inputs["response"]
                    #                     )

                    # print(inputs["response_input"])
                    # print(inputs["response"])
                    outputs = self.model(encoder_outputs=outputs[-1:], #skip loss and logits
                                        attention_mask=inputs["masks"],
                                        decoder_input_ids=inputs["response_input"],
                                        lm_labels=inputs["response"]
                                        )
                    resp_loss = outputs[0]

                    py_prev['bspn'] = turn_batch['bspn']

                    total_loss = (dst_loss + resp_loss) / self.args.gradient_accumulation_steps

                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                    if step % self.args.gradient_accumulation_steps == 0:
                        self.optim.step()
                        self.optim.zero_grad()
                    step+=1
                    log_loss += float(total_loss.item())
                    log_dst +=float(dst_loss.item())
                    log_resp +=float(resp_loss.item())
                    log_cnt += 1

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [total|bspn|resp] loss: {:.2f} {:.2f} {:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                           log_loss/(log_cnt+ 1e-8),
                                                                           log_dst/(log_cnt+ 1e-8),log_resp/(log_cnt+ 1e-8),
                                                                           time.time()-btm,
                                                                           turn_num+1))
            epoch_sup_loss = log_loss/(log_cnt+ 1e-8)
            do_test = False
            valid_loss = self.validate(do_test=do_test)
            logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))

            if valid_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                prev_min_loss = valid_loss
                torch.save(self.model.state_dict(), os.path.join(self.args.model_path, WEIGHTS_NAME))
                logging.info('Model saved')
                #self.save_model(epoch)
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
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
                # if not weight_decay_count:
                #     self.optim = AdamW(model.parameters(), lr=args.lr)
                #     lr *= cfg.lr_decay
                #     self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                #                   weight_decay=5e-5)
                #     weight_decay_count = cfg.weight_decay_count
                #     logging.info('learning rate decay, learning rate: %f' % (lr))

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
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(self.args.device)
                if self.args.noupdate_dst:
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                else:
                    dst_outputs, resp_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                turn_batch['resp_gen'] = resp_outputs
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs

            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, _ = self.reader.wrap_result(result_collection)
        bleu, success, match = self.evaluator.validation_metric(results)
        score = 0.5 * (success + match) + bleu
        valid_loss = 130 - score
        logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
        self.model.train()
        if do_test:
            print('result preview...')
            self.eval()
        return valid_loss

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
                    if k!="turn_domain":
                        inputs[k] = inputs[k].to(self.args.device)
                if self.args.noupdate_dst:
                    dst_outputs, resp_outputs = self.model.inference_sequicity(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                else:
                    dst_outputs, resp_outputs = self.model.inference(tokenizer=self.tokenizer, reader=self.reader, prev=py_prev, input_ids=inputs['input_ids'],attention_mask=inputs["masks"], turn_domain=inputs["turn_domain"], db=inputs["input_pointer"])
                turn_batch['resp_gen'] = resp_outputs
                turn_batch['bspn_gen'] = dst_outputs
                py_prev['bspn'] = dst_outputs
             
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        results, field = self.reader.wrap_result(result_collection)

        self.reader.save_result('w', results, field)

        metric_results = self.evaluator.run_metrics(results, eval_act=False)
        metric_field = list(metric_results[0].keys())
        req_slots_acc = metric_results[0]['req_slots_acc']
        info_slots_acc = metric_results[0]['info_slots_acc']

        self.reader.save_result('w', metric_results, metric_field,
                                            write_title='EVALUATION RESULTS:')
        self.reader.save_result('a', [info_slots_acc], list(info_slots_acc.keys()),
                                            write_title='INFORM ACCURACY OF EACH SLOTS:')
        self.reader.save_result('a', [req_slots_acc], list(req_slots_acc.keys()),
                                            write_title='REQUEST SUCCESS RESULTS:')
        self.reader.save_result('a', results, field+['wrong_domain', 'wrong_act', 'wrong_inform'],
                                            write_title='DECODED RESULTS:')
        self.reader.save_result_report(metric_results)

        # self.reader.metric_record(metric_results)
        self.model.train()
        return None

    def lexicalize(self, result_path,output_path):
        self.reader.relex(result_path,output_path)

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True


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
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')

    parser = argparse.ArgumentParser()
    parser.add_argument('--mode')
    parser.add_argument('--cfg', nargs='*')
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device (cuda or cpu)")
    parser.add_argument("--max_norm", type=float, default=1.0, help="Clipping gradient norm")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Accumulate gradients on several steps")
    parser.add_argument("--pretrained_checkpoint", type=str, default="t5-small", help="t5-small, t5-base, bart-large")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--context_window", type=int, default=5, help="how many previous turns for model input")
    parser.add_argument("--lr_decay", type=float, default=0.8, help="Learning rate decay")
    parser.add_argument("--noupdate_dst", action='store_true', help="dont use update base DST")
    parser.add_argument("--back_bone", type=str, default="t5", help="choose t5 or bart")
    #parser.add_argument("--dst_weight", type=int, default=1)
    parser.add_argument("--fraction", type=float, default=1.0)
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode == 'relex':
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
            args.model_path = 'experiments/{}_sd{}_lr{}_bs{}_sp{}_dc{}_cw{}_model_{}_noupdate{}_{}/'.format('-'.join(cfg.exp_domains), cfg.seed, args.lr, cfg.batch_size,
                                                                                            cfg.early_stop_count, args.lr_decay, args.context_window, args.pretrained_checkpoint, args.noupdate_dst, args.fraction)
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
    elif args.mode == 'relex':
        m = Model(args,test=True)
        output_path = os.path.join(args.model_path, 'generation.csv')
        m.lexicalize(cfg.result_path,output_path)


if __name__ == '__main__':
    main()
