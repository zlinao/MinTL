# checking all domain slots
import json
from damd_multiwoz.ontology import informable_slots


ONTOLOGY_PATH = './db/ontology.json'


def check_slots():
    onto_file = open(ONTOLOGY_PATH, 'r', encoding='utf8')
    onto_data = json.loads(onto_file.read())  # dict

    print('Checking all domain in ontology.json ...')
    for key, val in informable_slots.items():
        ontology = [f'{key}-{x}' for x in val]
        for onto in ontology:
            if onto not in onto_data.keys():
                print(f'Missing slot [{onto}]')
    print('Checking complete')


if __name__ == '__main__':
    check_slots()
