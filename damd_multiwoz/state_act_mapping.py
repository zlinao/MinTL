import json, zipfile
from reader import MultiWozReader
from collections import OrderedDict
import ontology


def dialog_turn_state_analysis(mode='train'):
    data_path = 'data/multi-woz-processed/data_for_damd.json'
    conv_data = 'data/multi-woz/annotated_user_da_with_span_full.json'
    archive = zipfile.ZipFile(conv_data + '.zip', 'r')
    convlab_data = json.loads(archive.open(conv_data.split('/')[-1], 'r').read().lower())
    reader = MultiWozReader()
    data = json.loads(open(data_path, 'r', encoding='utf-8').read().lower())

    turn_state_record, turn_state_count, golden_acts = {}, {}, {}
    act_state_collect = []
    act_state_detail = {}
    state_valid_acts = {}
    dial_count = 0
    turn_count = 0

    for fn, dial in data.items():
        dial_count += 1
        state_valid_acts[fn] = {}
        for turn_no, turn in enumerate(dial['log']):
            turn_state = {}
            turn_domain = turn['turn_domain'].split()
            cons_delex = turn['cons_delex'].split()
            sys_act = turn['sys_act']
            usr_act = convlab_data[fn]['log'][turn_no * 2]['dialog_act']
            db_ptr = [int(i) for i in turn['pointer'].split(',')]
            match = turn['match']
            if len(turn_domain) != 1 or turn_domain[0] == '[general]' or not sys_act:
                continue
            state_valid_acts[fn][turn_no] = {}
            turn_count += 1

            slot_mentioned = []
            for idx, tk in enumerate(cons_delex[:-1]):
                if tk in turn_domain:
                    i = idx+1
                    while i < len(cons_delex):
                        if '[' not in cons_delex[i]:
                            slot_mentioned.append(cons_delex[i])
                        else:
                            break
                        i = i+1
            slot_mentioned.sort()
            # turn_state['slot_mentioned'] = len(slot_mentioned)
            turn_state['domain'] = turn_domain
            turn_state['slot_mentioned'] = slot_mentioned
            if match == '':
                turn_state['match']=''
            elif match == '0':
                turn_state['match']='0'
            elif match == '1':
                turn_state['match'] = '1'
            elif match == '2' or match == '3':
                turn_state['match'] = '2-3'
            else:
                turn_state['match']='>3'
            if db_ptr[-2:] == [0,0]:
                turn_state['book'] = ''
            elif db_ptr[-2:] == [1,0]:
                turn_state['book'] = 'no'
            else:
                turn_state['book'] = 'yes'

            turn_state['usract'] = []
            for act in usr_act:
                d, a = act.split('-')
                if a not in turn_state['usract']:
                    slot_list = []
                    if a == 'request':
                        for slot_value in usr_act[act]:

                            slot = slot_value[0]

                            if slot == 'none':
                                continue
                            elif slot not in slot_list:
                                slot = ontology.da_abbr_to_slot_name.get(slot, slot)
                                slot_list.append(slot)
                    if not slot_list:
                        turn_state['usract'].append(a)
                    else:
                        slot_list.sort()
                        turn_state['usract'].append(a+'('+','.join(slot_list)+')')
            turn_state['usract'].sort()

            turn_state_str = ''
            for k,v in turn_state.items():
                if isinstance(v, list):
                    v_ = ','.join(v)
                elif isinstance(v, int):
                    v_ = str(v)
                else:
                    v_ = v
                turn_state_str += '%s(%s);'%(k, v_)
            turn_state_str = turn_state_str[:-1]
            state_valid_acts[fn][turn_no]['usdx'] = turn['user_delex']
            state_valid_acts[fn][turn_no]['state'] = turn_state_str


            if sys_act not in act_state_detail:
                act_state_detail[sys_act] = 1
            act_list = reader.aspan_to_act_list(sys_act)
            act_state = {'domain': {}, 'general': {}}
            for act in act_list:
                d, a, s = act.split('-')
                if d == 'general':
                    act_state['general'][a] = ''
                else:
                    if a not in act_state['domain']:
                        if s != 'none':
                            act_state['domain'][a] = ''
                        else:
                            act_state['domain'][a] = ''
                    else:
                        act_state['domain'][a] = ''

            no_order_act = {}
            for a in act_list:
                no_order_act[a] = 1

            act_state_str = ''
            for k,v in act_state.items():
                if isinstance(v, dict):
                    v_ = ''
                    for kk, vv in v.items():
                        v_ += kk+'(%s),'%str(vv)
                    if v_.endswith(','):
                        v_ = v_[:-1]
                elif isinstance(v, int):
                    v_ = str(v)
                else:
                    v_ = v
                if v_ != '':
                    act_state_str += '%s(%s);'%(k, v_)
            act_state_str = act_state_str[:-1]
            state_valid_acts[fn][turn_no]['gold'] = {}
            state_valid_acts[fn][turn_no]['gold'][act_state_str] = {}
            state_valid_acts[fn][turn_no]['gold'][act_state_str]['resp'] = turn['resp']
            state_valid_acts[fn][turn_no]['gold'][act_state_str]['act'] = sys_act

            if mode == 'test' and fn not in reader.test_files:
                continue
            if mode == 'train' and fn in reader.test_files:
                continue
            if act_state not in act_state_collect:
                act_state_collect.append(act_state)
            new_state = True if turn_state_str not in turn_state_record else False
            raw_sys_rec  = fn+'-'+str(turn_no)+':'+sys_act
            if new_state:
                turn_state_record[turn_state_str] = {act_state_str: {'num': 1, 'raw_acts': [raw_sys_rec], 'no_order_act': [no_order_act],
                                                                         'user': [turn['user']], 'resp': [turn['resp']]}}
                golden_acts[turn_state_str] = {'act_span': raw_sys_rec, 'no_order_act': no_order_act}
                turn_state_count[turn_state_str] = 1
            else:
                turn_state_count[turn_state_str] += 1
                if act_state_str in turn_state_record[turn_state_str]:
                    if no_order_act == golden_acts[turn_state_str]['no_order_act']:
                        continue
                    if no_order_act in turn_state_record[turn_state_str][act_state_str]['no_order_act']:
                        continue
                    turn_state_record[turn_state_str][act_state_str]['num'] +=1
                    turn_state_record[turn_state_str][act_state_str]['raw_acts'].append(raw_sys_rec)
                    turn_state_record[turn_state_str][act_state_str]['user'].append(turn['user'])
                    turn_state_record[turn_state_str][act_state_str]['resp'].append(turn['resp'])
                    turn_state_record[turn_state_str][act_state_str]['no_order_act'].append(no_order_act)
                else:
                    turn_state_record[turn_state_str][act_state_str] = {'num': 1, 'raw_acts': [raw_sys_rec], 'no_order_act': [no_order_act],
                                                                                                    'user': [turn['user']], 'resp': [turn['resp']]}
    for state, acts in turn_state_record.items():
        turn_state_record[state] = OrderedDict(sorted(acts.items(), key=lambda i:i[1]['num'], reverse=True))

    # print(mode)
    print('dialog count:', dial_count, 'turn count: ',turn_count)
    print('state count:', len(turn_state_record))
    print('raw act span count:', len(act_state_detail))
    print('act state count:', len(act_state_collect))


    for fn, dial in data.items():
        if fn in reader.dev_files or fn in reader.test_files:
            continue
        dial_count += 1
        for turn_no, turn in enumerate(dial['log']):
            if turn_no not in state_valid_acts[fn]:
                continue
            state = state_valid_acts[fn][turn_no]['state']
            gold_act_type = list(state_valid_acts[fn][turn_no]['gold'].keys())[0]
            state_valid_acts[fn][turn_no]['other'] = {}
            if state in turn_state_record:
                for act_type in turn_state_record[state]:
                    if act_type == gold_act_type:
                        continue
                    state_valid_acts[fn][turn_no]['other'][act_type] = []
                    for idx, a in enumerate(turn_state_record[state][act_type]['raw_acts']):
                        m = {'act': a}
                        m['resp'] = turn_state_record[state][act_type]['resp'][idx]
                        state_valid_acts[fn][turn_no]['other'][act_type].append(m)

    # sub_state_valid_acts = {}
    # count = 0
    # for fn, dial in state_valid_acts.items():
    #     if 'mul' in fn and fn not in reader.test_files and count<=100:
    #         sub_state_valid_acts[fn] = dial
    #         count += 1
    #     if count >100:
    #         break
    # with open('data/multi-woz-processed/example_multi_act_dialogs.json', 'w') as f:
    #     json.dump(sub_state_valid_acts, f, indent=2)

    idx_save = {}
    act_span_save = {}
    hist = []
    for fn, dial in state_valid_acts.items():
        if fn in reader.dev_files or fn in reader.test_files:
            continue
        act_span_save[fn] = {}
        idx_save[fn] = {}
        for turn_num, turn in dial.items():
            act_span_save[fn][turn_num] = {}
            idx_save[fn][turn_num] = []
            for act_type, acts in turn['other'].items():
                hist.append(len(acts)+1)
                act_span_save[fn][turn_num][act_type] = [a['act'].split(':')[1] for a in acts]
                idx_save[fn][turn_num].append([a['act'].split(':')[0] for a in acts])


    with open('data/multi-woz-processed/multi_act_mapping_%s.json'%mode, 'w') as f:
        json.dump(act_span_save, f, indent=2)



if __name__ == '__main__':
    dialog_turn_state_analysis()