import os, random, argparse, time, logging, json, tqdm
import numpy as np

import torch
from torch.optim import Adam

import utils
from config import global_config as cfg
from reader import MultiWozReader
from damd_net import DAMD, cuda_, get_one_hot_input
from eval import MultiWozEvaluator


class Model(object):
    def __init__(self):
        self.reader = MultiWozReader()
        if len(cfg.cuda_device)==1:
            self.m =DAMD(self.reader)
        else:
            m = DAMD(self.reader)
            self.m=torch.nn.DataParallel(m, device_ids=cfg.cuda_device)
            # print(self.m.module)
        self.evaluator = MultiWozEvaluator(self.reader) # evaluator class
        if cfg.cuda: self.m = self.m.cuda()  #cfg.cuda_device[0]
        self.optim = Adam(lr=cfg.lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),weight_decay=5e-5)
        self.base_epoch = -1

        if cfg.limit_bspn_vocab:
            self.reader.bspn_masks_tensor = {}
            for key, values in self.reader.bspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.bspn_masks_tensor[key] = v_
        if cfg.limit_aspn_vocab:
            self.reader.aspn_masks_tensor = {}
            for key, values in self.reader.aspn_masks.items():
                v_ = cuda_(torch.Tensor(values).long())
                self.reader.aspn_masks_tensor[key] = v_

    def add_torch_input(self, inputs, mode='train', first_turn=False):
        need_onehot = ['user', 'usdx', 'bspn', 'aspn', 'pv_resp', 'pv_bspn', 'pv_aspn',
                                   'dspn', 'pv_dspn', 'bsdx', 'pv_bsdx']
        inputs['db'] = cuda_(torch.from_numpy(inputs['db_np']).float())
        for item in ['user', 'usdx', 'resp', 'bspn', 'aspn', 'bsdx', 'dspn']:
            if not cfg.enable_aspn and item == 'aspn':
                continue
            if not cfg.enable_bspn and item == 'bspn':
                continue
            if not cfg.enable_dspn and item == 'dspn':
                continue
            inputs[item] = cuda_(torch.from_numpy(inputs[item+'_unk_np']).long())
            if item in ['user', 'usdx', 'resp', 'bspn']:
                inputs[item+'_nounk'] = cuda_(torch.from_numpy(inputs[item+'_np']).long())
            else:
                inputs[item+'_nounk'] = inputs[item]
            # print(item, inputs[item].size())
            if item in ['resp', 'bspn', 'aspn', 'bsdx', 'dspn']:
                if 'pv_'+item+'_unk_np' not in inputs:
                    continue
                inputs['pv_'+item] = cuda_(torch.from_numpy(inputs['pv_'+item+'_unk_np']).long())
                if item in ['user', 'usdx', 'bspn']:
                    inputs['pv_'+item+'_nounk'] = cuda_(torch.from_numpy(inputs['pv_'+item+'_np']).long())
                    inputs[item+'_4loss'] = self.index_for_loss(item, inputs)
                else:
                    inputs['pv_'+item+'_nounk'] = inputs['pv_'+item]
                    inputs[item+'_4loss'] = inputs[item]
                if 'pv_' + item in need_onehot:
                    inputs['pv_' + item + '_onehot'] = get_one_hot_input(inputs['pv_'+item+'_unk_np'])
            if item in need_onehot:
                inputs[item+'_onehot'] = get_one_hot_input(inputs[item+'_unk_np'])

        if cfg.multi_acts_training and 'aspn_aug_unk_np' in inputs:
            inputs['aspn_aug'] = cuda_(torch.from_numpy(inputs['aspn_aug_unk_np']).long())
            inputs['aspn_aug_4loss'] = inputs['aspn_aug']

        return inputs

    def index_for_loss(self, item, inputs):
        raw_labels = inputs[item+'_np']
        if item == 'bspn':
            copy_sources = [inputs['user_np'], inputs['pv_resp_np'], inputs['pv_bspn_np']]
        elif item == 'bsdx':
            copy_sources = [inputs['usdx_np'], inputs['pv_resp_np'], inputs['pv_bsdx_np']]
        elif item == 'aspn':
            copy_sources = []
            if cfg.use_pvaspn:
                copy_sources.append(inputs['pv_aspn_np'])
            if cfg.enable_bspn:
                copy_sources.append(inputs[cfg.bspn_mode+'_np'])
        elif item == 'dspn':
            copy_sources = [inputs['pv_dspn_np']]
        elif item == 'resp':
            copy_sources = [inputs['usdx_np']]
            if cfg.enable_bspn:
                copy_sources.append(inputs[cfg.bspn_mode+'_np'])
            if cfg.enable_aspn:
                copy_sources.append(inputs['aspn_np'])
        else:
            return
        new_labels = np.copy(raw_labels)
        if copy_sources:
            bidx, tidx = np.where(raw_labels>=self.reader.vocab_size)
            copy_sources = np.concatenate(copy_sources, axis=1)
            for b in bidx:
                for t in tidx:
                    oov_idx = raw_labels[b, t]
                    if len(np.where(copy_sources[b, :] == oov_idx)[0])==0:
                        new_labels[b, t] = 2
        return cuda_(torch.from_numpy(new_labels).long())

    def train(self):
        lr = cfg.lr
        prev_min_loss, early_stop_count = 1 << 30, cfg.early_stop_count
        weight_decay_count = cfg.weight_decay_count
        train_time = 0
        sw = time.time()

        for epoch in range(cfg.epoch_num):
            if epoch <= self.base_epoch:
                continue
            self.training_adjust(epoch)
            sup_loss = 0
            sup_cnt = 0
            optim = self.optim
            # data_iterator generatation size: (batch num, turn num, batch size)
            btm = time.time()
            data_iterator = self.reader.get_batches('train')
            for iter_num, dial_batch in enumerate(data_iterator):
                hidden_states = {}
                py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
                bgt = time.time()
                for turn_num, turn_batch in enumerate(dial_batch):
                    # print('turn %d'%turn_num)
                    # print(len(turn_batch['dial_id']))
                    optim.zero_grad()
                    first_turn = (turn_num==0)
                    inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                    # for k,v in inputs.items():
                    #     if k in ["turn_domain",'db_np']:
                    #         print(f"{k}: {v[0]}")
                    #     else:
                    #         print(f"{k}: {self.reader.vocab.sentence_decode(v[0])}")
                    # if turn_num==1:
                    #     exit(0)
                    inputs = self.add_torch_input(inputs, first_turn=first_turn)
                    # total_loss, losses, hidden_states = self.m(inputs, hidden_states, first_turn, mode='train')
                    total_loss, losses = self.m(inputs, hidden_states, first_turn, mode='train')
                    # print('forward completed')
                    py_prev['pv_resp'] = turn_batch['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn']

                    total_loss = total_loss.mean()
                    # print('forward time:%f'%(time.time()-test_begin))
                    # test_begin = time.time()
                    total_loss.backward(retain_graph=False)
                    # total_loss.backward(retain_graph=turn_num != len(dial_batch) - 1)
                    # print('backward time:%f'%(time.time()-test_begin))
                    grad = torch.nn.utils.clip_grad_norm_(self.m.parameters(), 5.0)
                    optim.step()
                    sup_loss += float(total_loss)
                    sup_cnt += 1
                    torch.cuda.empty_cache()

                if (iter_num+1)%cfg.report_interval==0:
                    logging.info(
                            'iter:{} [total|bspn|aspn|resp] loss: {:.2f} {:.2f} {:.2f} {:.2f} grad:{:.2f} time: {:.1f} turn:{} '.format(iter_num+1,
                                                                           float(total_loss),
                                                                           float(losses[cfg.bspn_mode]),float(losses['aspn']),float(losses['resp']),
                                                                           grad,
                                                                           time.time()-btm,
                                                                           turn_num+1))
                    if cfg.enable_dst and cfg.bspn_mode == 'bsdx':
                        logging.info('bspn-dst:{:.3f}'.format(float(losses['bspn'])))
                    if cfg.multi_acts_training:
                        logging.info('aspn-aug:{:.3f}'.format(float(losses['aspn_aug'])))

                # btm = time.time()
                # if (iter_num+1)%40==0:
                #     print('validation checking ... ')
                #     valid_sup_loss, valid_unsup_loss = self.validate(do_test=False)
                #     logging.info('validation loss in epoch %d sup:%f unsup:%f' % (epoch, valid_sup_loss, valid_unsup_loss))

            epoch_sup_loss = sup_loss / (sup_cnt + 1e-8)
            # do_test = True if (epoch+1)%5==0 else False
            do_test = False
            valid_loss = self.validate(do_test=do_test)
            logging.info('epoch: %d, train loss: %.3f, valid loss: %.3f, total time: %.1fmin' % (epoch+1, epoch_sup_loss,
                    valid_loss, (time.time()-sw)/60))
            # self.save_model(epoch)
            if valid_loss <= prev_min_loss:
                early_stop_count = cfg.early_stop_count
                weight_decay_count = cfg.weight_decay_count
                prev_min_loss = valid_loss
                self.save_model(epoch)
            else:
                early_stop_count -= 1
                weight_decay_count -= 1
                logging.info('epoch: %d early stop countdown %d' % (epoch+1, early_stop_count))
                if not early_stop_count:
                    self.load_model()
                    print('result preview...')
                    file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log%s.json'%cfg.seed))
                    logging.getLogger('').addHandler(file_handler)
                    logging.info(str(cfg))
                    self.eval()
                    return
                if not weight_decay_count:
                    lr *= cfg.lr_decay
                    self.optim = Adam(lr=lr, params=filter(lambda x: x.requires_grad, self.m.parameters()),
                                  weight_decay=5e-5)
                    weight_decay_count = cfg.weight_decay_count
                    logging.info('learning rate decay, learning rate: %f' % (lr))
        self.load_model()
        print('result preview...')
        file_handler = logging.FileHandler(os.path.join(cfg.exp_path, 'eval_log%s.json'%cfg.seed))
        logging.getLogger('').addHandler(file_handler)
        logging.info(str(cfg))
        self.eval()


    def validate(self, data='dev', do_test=False):
        self.m.eval()
        valid_loss, count = 0, 0
        data_iterator = self.reader.get_batches(data)
        result_collection = {}
        for batch_num, dial_batch in enumerate(data_iterator):
            hidden_states = {}
            py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx': None}
            for turn_num, turn_batch in enumerate(dial_batch):
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                inputs = self.add_torch_input(inputs, first_turn=first_turn)
                # total_loss, losses, hidden_states = self.m(inputs, hidden_states, first_turn, mode='train')
                if cfg.valid_loss not in ['score', 'match', 'success', 'bleu']:
                    total_loss, losses = self.m(inputs, hidden_states, first_turn, mode='train')
                    py_prev['pv_resp'] = turn_batch['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_bspn'] = turn_batch['bspn']
                        py_prev['pv_bsdx'] = turn_batch['bsdx']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn']

                    if cfg.valid_loss == 'total_loss':
                        valid_loss += float(total_loss)
                    elif cfg.valid_loss == 'bspn_loss':
                        valid_loss += float(losses[cfg.bspn_mode])
                    elif cfg.valid_loss == 'aspn_loss':
                        valid_loss += float(losses['aspn'])
                    elif cfg.valid_loss == 'resp_loss':
                        valid_loss += float(losses['reps'])
                    else:
                        raise ValueError('Invalid validation loss type!')
                else:
                    decoded = self.m(inputs, hidden_states, first_turn, mode='test')
                    turn_batch['resp_gen'] = decoded['resp']
                    if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
                        turn_batch['bspn_gen'] = decoded['bspn']
                    py_prev['pv_resp'] = turn_batch['resp'] if cfg.use_true_pv_resp else decoded['resp']
                    if cfg.enable_bspn:
                        py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode]
                        py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                    if cfg.enable_aspn:
                        py_prev['pv_aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                    if cfg.enable_dspn:
                        py_prev['pv_dspn'] = turn_batch['dspn'] if cfg.use_true_prev_dspn else decoded['dspn']
                count += 1
                torch.cuda.empty_cache()

            if cfg.valid_loss in ['score', 'match', 'success', 'bleu']:
                result_collection.update(self.reader.inverse_transpose_batch(dial_batch))


        if cfg.valid_loss not in ['score', 'match', 'success', 'bleu']:
            valid_loss /= (count + 1e-8)
        else:
            results, _ = self.reader.wrap_result(result_collection)
            bleu, success, match = self.evaluator.validation_metric(results)
            score = 0.5 * (success + match) + bleu
            valid_loss = 130 - score
            logging.info('validation [CTR] match: %2.1f  success: %2.1f  bleu: %2.1f'%(match, success, bleu))
        self.m.train()
        if do_test:
            print('result preview...')
            self.eval()
        return valid_loss

    def eval(self, data='test'):
        self.m.eval()
        self.reader.result_file = None
        result_collection = {}
        data_iterator = self.reader.get_batches(data)
        for batch_num, dial_batch in tqdm.tqdm(enumerate(data_iterator)):
            # quit()
            # if batch_num > 0:
            #     continue
            hidden_states = {}
            py_prev = {'pv_resp': None, 'pv_bspn': None, 'pv_aspn':None, 'pv_dspn': None, 'pv_bsdx':None}
            print('batch_size:', len(dial_batch[0]['resp']))
            for turn_num, turn_batch in enumerate(dial_batch):
                # print('turn %d'%turn_num)
                # if turn_num!=0 and turn_num<4:
                #     continue
                first_turn = (turn_num==0)
                inputs = self.reader.convert_batch(turn_batch, py_prev, first_turn=first_turn)
                inputs = self.add_torch_input(inputs, first_turn=first_turn)
                decoded = self.m(inputs, hidden_states, first_turn, mode='test')
                #print(decoded)
                turn_batch['resp_gen'] = decoded['resp']
                if cfg.bspn_mode == 'bsdx':
                    turn_batch['bsdx_gen'] = decoded['bsdx'] if cfg.enable_bspn else [[0]] * len(decoded['resp'])
                if cfg.bspn_mode == 'bspn' or cfg.enable_dst:
                    turn_batch['bspn_gen'] = decoded['bspn'] if cfg.enable_bspn else [[0]] * len(decoded['resp'])
                turn_batch['aspn_gen'] = decoded['aspn'] if cfg.enable_aspn else [[0]] * len(decoded['resp'])
                turn_batch['dspn_gen'] = decoded['dspn'] if cfg.enable_dspn else [[0]] * len(decoded['resp'])

                if self.reader.multi_acts_record is not None:
                    turn_batch['multi_act_gen'] = self.reader.multi_acts_record
                if cfg.record_mode:
                    turn_batch['multi_act'] = self.reader.aspn_collect
                    turn_batch['multi_resp'] = self.reader.resp_collect
                # print(turn_batch['user'])
                # print('user:', self.reader.vocab.sentence_decode(turn_batch['user'][0] , eos='<eos_u>', indicate_oov=True))
                # print('resp:', self.reader.vocab.sentence_decode(decoded['resp'][0] , eos='<eos_r>', indicate_oov=True))
                # print('bspn:', self.reader.vocab.sentence_decode(decoded['bspn'][0] , eos='<eos_b>', indicate_oov=True))
                # for b in range(len(decoded['resp'])):
                #     for i in range(5):
                #         print('aspn:', self.reader.vocab.sentence_decode(decoded['aspn'][i][b] , eos='<eos_a>', indicate_oov=True))

                py_prev['pv_resp'] = turn_batch['resp'] if cfg.use_true_pv_resp else decoded['resp']
                if cfg.enable_bspn:
                    py_prev['pv_'+cfg.bspn_mode] = turn_batch[cfg.bspn_mode] if cfg.use_true_prev_bspn else decoded[cfg.bspn_mode]
                    py_prev['pv_bspn'] = turn_batch['bspn'] if cfg.use_true_prev_bspn or 'bspn' not in decoded else decoded['bspn']
                if cfg.enable_aspn:
                    py_prev['pv_aspn'] = turn_batch['aspn'] if cfg.use_true_prev_aspn else decoded['aspn']
                if cfg.enable_dspn:
                    py_prev['pv_dspn'] = turn_batch['dspn'] if cfg.use_true_prev_dspn else decoded['dspn']
                torch.cuda.empty_cache()
                # prev_z = turn_batch['bspan']
            # print('test iter %d'%(batch_num+1))
            # print(dial_batch)
            result_collection.update(self.reader.inverse_transpose_batch(dial_batch))

        # self.reader.result_file.close()
        if cfg.record_mode:
            self.reader.record_utterance(result_collection)
            quit()
        
        results, field = self.reader.wrap_result(result_collection)
        #print(results)
        self.reader.save_result('w', results, field)

        metric_results = self.evaluator.run_metrics(results)
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
        self.m.train()
        return None

    def save_model(self, epoch, path=None, critical=False):
        if not cfg.save_log:
            return
        if not path:
            path = cfg.model_path
        if critical:
            path += '.final'
        all_state = {'lstd': self.m.state_dict(),
                     'config': cfg.__dict__,
                     'epoch': epoch}
        torch.save(all_state, path)
        logging.info('Model saved')

    def load_model(self, path=None):
        if not path:
            path = cfg.model_path
        all_state = torch.load(path, map_location='cpu')
        self.m.load_state_dict(all_state['lstd'])
        self.base_epoch = all_state.get('epoch', 0)
        logging.info('Model loaded')

    def training_adjust(self, epoch):
        return

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def load_glove_embedding(self, freeze=False):
        if not cfg.multi_gpu:
            initial_arr = self.m.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.embedding.weight.data.copy_(emb)
        else:
            initial_arr = self.m.module.embedding.weight.data.cpu().numpy()
            emb = torch.from_numpy(utils.get_glove_matrix(
                            cfg.glove_path, self.reader.vocab, initial_arr))
            self.m.module.embedding.weight.data.copy_(emb)


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
    parser.add_argument('-mode')
    parser.add_argument('-cfg', nargs='*')
    args = parser.parse_args()

    cfg.mode = args.mode
    if args.mode == 'test' or args.mode=='adjust':
        parse_arg_cfg(args)
        cfg_load = json.loads(open(os.path.join(cfg.eval_load_path, 'config.json'), 'r').read())
        for k, v in cfg_load.items():
            if k in ['mode', 'cuda', 'cuda_device', 'eval_load_path', 'eval_per_domain', 'use_true_pv_resp',
                        'use_true_prev_bspn','use_true_prev_aspn','use_true_curr_bspn','use_true_curr_aspn',
                        'name_slot_unable', 'book_slot_unable','count_req_dials_only','log_time', 'model_path',
                        'result_path', 'model_parameters', 'multi_gpu', 'use_true_bspn_for_ctr_eval', 'nbest',
                        'limit_bspn_vocab', 'limit_aspn_vocab', 'same_eval_as_cambridge', 'beam_width',
                        'use_true_domain_for_ctr_eval', 'use_true_prev_dspn', 'aspn_decode_mode',
                        'beam_diverse_param', 'same_eval_act_f1_as_hdsa', 'topk_num', 'nucleur_p',
                        'act_selection_scheme', 'beam_penalty_type', 'record_mode']:
                continue
            setattr(cfg, k, v)
            cfg.model_path = os.path.join(cfg.eval_load_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.eval_load_path, 'result.csv')
    else:
        parse_arg_cfg(args)
        if cfg.exp_path in ['' , 'to be generated']:
            cfg.exp_path = 'experiments/{}_{}_sd{}_lr{}_bs{}_sp{}_dc{}_act{}_0.05/'.format('-'.join(cfg.exp_domains),
                                                                                            cfg.exp_no, cfg.seed, cfg.lr, cfg.batch_size,
                                                                                            cfg.early_stop_count, cfg.weight_decay_count, cfg.enable_aspn)
            if cfg.save_log:
                os.mkdir(cfg.exp_path)
            cfg.model_path = os.path.join(cfg.exp_path, 'model.pkl')
            cfg.result_path = os.path.join(cfg.exp_path, 'result.csv')
            cfg.vocab_path_eval = os.path.join(cfg.exp_path, 'vocab')
            cfg.eval_load_path = cfg.exp_path

    cfg._init_logging_handler(args.mode)
    if cfg.cuda:
        if len(cfg.cuda_device)==1:
            cfg.multi_gpu = False
            torch.cuda.set_device(cfg.cuda_device[0])
        else:
            # cfg.batch_size *= len(cfg.cuda_device)
            cfg.multi_gpu = True
            torch.cuda.set_device(cfg.cuda_device[0])
        logging.info('Device: {}'.format(torch.cuda.current_device()))


    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    m = Model()
    cfg.model_parameters = m.count_params()
    logging.info(str(cfg))

    if args.mode == 'train':
        if cfg.save_log:
            # open(cfg.exp_path + 'config.json', 'w').write(str(cfg))
            m.reader.vocab.save_vocab(cfg.vocab_path_eval)
            with open(os.path.join(cfg.exp_path, 'config.json'), 'w') as f:
                json.dump(cfg.__dict__, f, indent=2)
        # m.load_glove_embedding()
        m.train()
    elif args.mode == 'adjust':
        m.load_model(cfg.model_path)
        m.train()
    elif args.mode == 'test':
        m.load_model(cfg.model_path)
        # m.train()
        m.eval(data='test')



if __name__ == '__main__':
    main()
