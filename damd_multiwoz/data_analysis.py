import os, json, copy, re, zipfile
from collections import OrderedDict
from ontology import all_domains


data_path = 'data/multi-woz/'
save_path = 'data/multi-woz-analysis/'
save_path_exp = 'data/multi-woz-processed/'
data_file = 'data.json'
domains = all_domains

def analysis():
    compressed_raw_data = {}
    goal_of_dials = {}
    req_slots = {}
    info_slots = {}
    dom_count = {}
    dom_fnlist = {}
    all_domain_specific_slots = set()
    for domain in domains:
        req_slots[domain] = []
        info_slots[domain] = []

    archive = zipfile.ZipFile(data_path+data_file+'.zip', 'r')
    data = archive.open(data_file, 'r').read().decode('utf-8').lower()
    ref_nos = list(set(re.findall(r'\"reference\"\: \"(\w+)\"', data)))
    data = json.loads(data)

    for fn, dial in data.items():
        goals = dial['goal']
        logs = dial['log']

        # get compressed_raw_data and goal_of_dials
        compressed_raw_data[fn] = {'goal': {}, 'log': []}
        goal_of_dials[fn] = {}
        for dom, goal in goals.items():
            if dom != 'topic' and dom != 'message' and goal:
                compressed_raw_data[fn]['goal'][dom] = goal
                goal_of_dials[fn][dom] = goal

        for turn in logs:
            if not turn['metadata']:
                compressed_raw_data[fn]['log'].append({'text': turn['text']})
            else:
                meta = turn['metadata']
                turn_dict = {'text': turn['text'], 'metadata': {}}
                for dom, book_semi in meta.items():
                    book, semi = book_semi['book'], book_semi['semi']
                    record = False
                    for slot, value in book.items():
                        if value not in ['', []]:
                            record = True
                    if record:
                        turn_dict['metadata'][dom] = {}
                        turn_dict['metadata'][dom]['book'] = book
                    record = False
                    for slot, value in semi.items():
                        if value not in ['', []]:
                            record = True
                            break
                    if record:
                        for s, v in copy.deepcopy(semi).items():
                            if v == 'not mentioned':
                                del semi[s]
                        if not turn_dict['metadata'].get(dom):
                            turn_dict['metadata'][dom] = {}
                        turn_dict['metadata'][dom]['semi'] = semi
                compressed_raw_data[fn]['log'].append(turn_dict)


            # get domain statistics
            dial_type = 'multi' if 'mul' in fn or 'MUL' in fn else 'single'
            if fn in ['pmul2756.json', 'pmul4958.json', 'pmul3599.json']:
                dial_type = 'single'
            dial_domains = [dom for dom in domains if goals[dom]]
            dom_str = ''
            for dom in dial_domains:
                if not dom_count.get(dom+'_'+dial_type):
                    dom_count[dom+'_'+dial_type] = 1
                else:
                    dom_count[dom+'_'+dial_type] += 1
                if not dom_fnlist.get(dom+'_'+dial_type):
                    dom_fnlist[dom+'_'+dial_type] = [fn]
                else:
                    dom_fnlist[dom+'_'+dial_type].append(fn)
                dom_str += '%s_'%dom
            dom_str = dom_str[:-1]
            if dial_type=='multi':
                if not dom_count.get(dom_str):
                    dom_count[dom_str] = 1
                else:
                    dom_count[dom_str] += 1
                if not dom_fnlist.get(dom_str):
                    dom_fnlist[dom_str] = [fn]
                else:
                    dom_fnlist[dom_str].append(fn)
            ######

            # get informable and requestable slots statistics
            for domain in domains:
                info_ss = goals[domain].get('info', {})
                book_ss = goals[domain].get('book', {})
                req_ss = goals[domain].get('reqt', {})
                for info_s in info_ss:
                    all_domain_specific_slots.add(domain+'-'+info_s)
                    if info_s not in info_slots[domain]:
                        info_slots[domain]+= [info_s]
                for book_s in book_ss:
                    if 'book_' + book_s not in info_slots[domain] and book_s not in ['invalid', 'pre_invalid']:
                        all_domain_specific_slots.add(domain+'-'+book_s)
                        info_slots[domain]+= ['book_' + book_s]
                for req_s in req_ss:
                    if req_s not in req_slots[domain]:
                        req_slots[domain]+= [req_s]



    # result statistics
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    if not os.path.exists(save_path_exp):
        os.mkdir(save_path_exp)
    with open(save_path+'req_slots.json', 'w') as sf:
        json.dump(req_slots,sf,indent=2)
    with open(save_path+'info_slots.json', 'w') as sf:
        json.dump(info_slots,sf,indent=2)
    with open(save_path+'all_domain_specific_info_slots.json', 'w') as sf:
        json.dump(list(all_domain_specific_slots),sf,indent=2)
        print("slot num:", len(list(all_domain_specific_slots)))
    with open(save_path+'goal_of_each_dials.json', 'w') as sf:
        json.dump(goal_of_dials, sf, indent=2)
    with open(save_path+'compressed_data.json', 'w') as sf:
        json.dump(compressed_raw_data, sf, indent=2)
    with open(save_path + 'domain_count.json', 'w') as sf:
        single_count = [d for d in dom_count.items() if 'single' in d[0]]
        multi_count = [d for d in dom_count.items() if 'multi' in d[0]]
        other_count = [d for d in dom_count.items() if 'multi' not in d[0] and 'single' not in d[0]]
        dom_count_od = OrderedDict(single_count+multi_count+other_count)
        json.dump(dom_count_od, sf, indent=2)
    with open(save_path_exp + 'reference_no.json', 'w') as sf:
        json.dump(ref_nos,sf,indent=2)
    with open(save_path_exp + 'domain_files.json', 'w') as sf:
        json.dump(dom_fnlist, sf, indent=2)


if __name__ == '__main__':
    analysis()