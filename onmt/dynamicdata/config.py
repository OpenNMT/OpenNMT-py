import collections
import itertools
import os
import yaml

import jsonschema

#def validate_schema(data_config):
#    # FIXME: use package resources
#    schema_path = 'data_config.schema.yaml'
#    with open(schema_path, 'r') as fobj:
#        schema = yaml.safe_load(fobj)
#    jsonschema.validate(data, schema)

def _inverse_groups(data_config):
    for input in data_config['inputs']:
        group = data_config['inputs'][input]['group']
        if '_inputs' not in data_config['groups'][group]:
            data_config['groups'][group]['_inputs'] = []
        data_config['groups'][group]['_inputs'].append(input)

def _normalize_sizes(data_config):
    size_per_group = collections.Counter()
    corpora_per_group = collections.Counter()
    for input in data_config['inputs']:
        group = data_config['inputs'][input]['group']
        if 'size' not in data_config['inputs'][input]:
            continue
        size = data_config['inputs'][input]['size']
        size_per_group[group] += size
        corpora_per_group[group] += 1
    for input in data_config['inputs']:
        group = data_config['inputs'][input]['group']
        if 'size' not in data_config['inputs'][input]:
            if corpora_per_group[group] == 0:
                # no sizes set
                data_config['inputs'][input]['size'] = 1
            else:
                # 
                data_config['inputs'][input]['size'] = int(
                    100 / corpora_per_group[group])
        else:
            size = data_config['inputs'][input]['size']
            data_config['inputs'][input]['size'] = int(
                100 * size / size_per_group[group])

def _all_transforms(data_config):
    all_transforms = set()
    for group in data_config['groups']:
        all_transforms.update(data_config['groups'][group].get('transforms', []))
    data_config['_transforms'] = list(sorted(all_transforms))

def _group_defaults(data_config):
    for group in data_config['groups']:
        if 'split' not in data_config['groups'][group]:
            data_config['groups'][group]['split'] = 'train'
        if 'weight' not in data_config['groups'][group]:
            data_config['groups'][group]['weight'] = 1

def read_data_config(data_config_file):
    with open(data_config_file, 'r') as fobj:
        data_config = yaml.safe_load(fobj)
    _inverse_groups(data_config)
    _normalize_sizes(data_config)
    _all_transforms(data_config)
    _group_defaults(data_config)
    #validate_schema(data_config)
    return data_config

def _filter_config(config, sub_config, path, rules):
    decision = 'continue'
    current = path[-1]
    for rule in rules:
        rule_path, rule_decision = rule
        if all(b is None or a == b for (a, b) in itertools.zip_longest(path, rule_path)):
            decision = rule_decision
            break
    if decision == 'keep':
        sub_config[current] = config[current]
    elif decision == 'drop':
        return
    else:
        if isinstance(config[current], dict):
            sub_config[current] = {}
            for key in config[current]:
                extended_path = path + [key]
                _filter_config(
                    config[current], sub_config[current], extended_path, rules)
        else:
            # default keep
            sub_config[current] = config[current]

def sharding_only(data_config):
    """ retains only config used in sharding step """
    rules = ((('meta', 'shard'), 'keep'),
             (('meta', 'train'), 'drop'),
             (('groups', None, 'transforms'), 'drop'),
             (('groups', None, 'weight'), 'drop'),
             (('groups', None, 'meta'), 'drop'),
             (('inputs',), 'keep'),
             (('_transforms',), 'drop'),
            )
    sub_config = {}
    for key in data_config:
        _filter_config(data_config, sub_config, [key], rules)
    return sub_config

def save_shard_config(data_config):
    stored_shard_config_file = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'stored_shard_config.yaml')
    if os.path.exists(stored_shard_config_file):
        raise Exception('stored shard config "{}"'
            ' already exists, not overwriting'.format(stored_shard_config_file))
    os.makedirs(
        data_config['meta']['shard']['rootdir'],
        exist_ok=True)
    with open(stored_shard_config_file, 'w') as fobj:
        yaml.safe_dump(data_config, fobj)

def verify_shard_config(data_config):
    stored_shard_config_file = os.path.join(
        data_config['meta']['shard']['rootdir'],
        'stored_shard_config.yaml')
    shard_config = sharding_only(data_config)
    with open(stored_shard_config_file, 'r') as fobj:
        stored_shard_config = yaml.safe_load(fobj)
    if not shard_config == stored_shard_config:
        raise Exception(
            'data_config not compatible with stored shard config.\n'
            'old {}\nnew {}\n'.format(stored_shard_config, shard_config))
