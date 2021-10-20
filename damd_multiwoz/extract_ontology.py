# ontology utilities
import os
import json
import argparse
from typing import Dict, List
from damd_multiwoz.ontology import all_domains


def get_parse():
    parser = argparse.ArgumentParser(description='Custom json input path.')
    parser.add_argument('--ontology', type=str,
                        default='./db/envi_ontology.json', help='raw ontology path')
    parser.add_argument('--goal_dialog', type=str,
                        default='./data/multi-woz-analysis/goal_of_each_dials.json', help='goal of each dials path')
    parser.add_argument('--save_path', type=str,
                        default='./db/', help='save extracted ontology and other dbs')

    return parser.parse_args()


def extract_ontology(path: str, save_path: str, goal_dialog_path: str = None):
    # extract from envi_ontology.json
    result_dict: Dict[str, List] = {}
    value_set: Dict[str, Dict] = {}

    avoid_key = ['bus']
    avoid_idx = ['hotel-price', 'hotel-location', 'attraction-location', 'restaurant-location']
    with open(path, 'r', encoding='utf8') as f:
        data = json.loads(f.read())
        keys_list = data.keys()
        print('Processing ontology ...')

        for main_key in keys_list:
            if main_key in avoid_key:
                continue
            value_set[main_key] = {}
            for item in data[main_key]:
                for sub_key, val in item['vn_entity'].items():
                    sub_key = str(sub_key).lower()
                    idx = f'{str(main_key).lower()}-{sub_key}'
                    if idx in avoid_idx:
                        continue
                    val = str(val).lower()
                    if idx not in result_dict:
                        result_dict[idx] = []
                    if sub_key not in value_set[main_key]:
                        value_set[main_key][sub_key] = []
                    if val not in result_dict[idx]:
                        result_dict[idx].append(val)
                        value_set[main_key][sub_key].append(val)

    print('Exporting ontology and value set ...')
    ontology_path = os.path.join(save_path, 'ontology.json')
    with open(ontology_path, 'w', encoding='utf8') as save_file:
        json.dump(result_dict, save_file, ensure_ascii=False)
        print('Extract ontology has been successfully')
    with open(os.path.join(save_path, 'value_set.json'), 'w', encoding='utf8') as save_file:
        json.dump(value_set, save_file, ensure_ascii=False)
        print('Extract value set has been successfully')

    _extract_ontology_2_db(path, save_path)


def _extract_ontology_2_db(path: str, save_path: str):
    with open(path, 'r', encoding='utf8') as f:
        data = json.loads(f.read())

        for main_key, val in data.items():
            db_list: List = []
            for item in val:
                db_list.append(item['vn_entity'])

            save_file_path = os.path.join(save_path, f'{main_key}_db.json')
            with open(save_file_path, 'w', encoding='utf8') as save_file:
                json.dump(db_list, save_file, ensure_ascii=False)
                print(f'Extracted [{main_key}] into {main_key}_db.json.')


def extract_ontology_from_goal_dialog(path: str, save_path: str):
    ontology_set: Dict[str, List] = {}  # key: [] for key in all_domains

    goal_of_dialog = open(path, 'r', encoding='utf8')

    dialog_data = json.loads(goal_of_dialog.read())
    for idx, dialog in dialog_data.items():
        for onto, onto_val in dialog.items():
            item_list = ['info', 'book'] if 'book' in onto_val.keys() else ['info']
            for item in item_list:
                for key, val in onto_val[item].items():
                    idx = f'{onto}-{key}'
                    if idx not in ontology_set:
                        ontology_set[idx] = []
                    if val not in ontology_set[idx]:
                        ontology_set[idx].append(val)

    goal_of_dialog.close()
    save_file_path = os.path.join(save_path, 'new_ontology.json')
    with open(save_file_path, 'w', encoding='utf8') as save_file:
        json.dump(ontology_set, save_file, ensure_ascii=False)
        print(f'Extracted ontology.json')


if __name__ == '__main__':
    opt = get_parse()

    # extract
    json_path = opt.ontology
    local_path = opt.save_path
    goal_path = opt.goal_dialog

    extract_ontology(json_path, local_path)
    # extract_ontology_from_goal_dialog(goal_path, local_path)
    # _extract_ontology_2_db(json_path, local_path)
