#!/usr/bin/env python
import argparse
import collections
import random
import torchtext
import yaml
from jinja2 import Environment

from onmt.dynamicdata.config import read_data_config, process_config, verify_shard_config, \
                                    sharding_only, remove_generated

def get_template(template_file):
    env = Environment()
    with open(template_file, 'r') as fobj:
        template = env.from_string(fobj.read())
    return template

def parse_vars(template_vars):
    pairs = [arg.split(':') for arg in template_vars]
    template_vars = {key: val for (key, val) in pairs}
    return template_vars

def template_main(shard_config_file, template_file, out_file, template_vars):
    shard_data_config = read_data_config(shard_config_file)
    shard_data_config = sharding_only(shard_data_config)

    template = get_template(template_file)
    rendered = template.render(**template_vars)
    data_config = yaml.safe_load(rendered)
    # sharding-related parts taken from shard conf
    data_config['meta']['shard'] = shard_data_config['meta']['shard']
    data_config['inputs'] = shard_data_config['inputs']
    data_config = process_config(data_config, template=True)
    verify_shard_config(data_config)
    data_config = remove_generated(data_config)
    with open(out_file, 'w') as fobj:
        yaml.safe_dump(data_config, fobj)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('template_vars', nargs='*', metavar='KEY:VAR',
                        help='template variables as key:value pairs')
    parser.add_argument('--shard_config_file', required=True)
    parser.add_argument('--template_file', required=True)
    parser.add_argument('--out_file', required=True)
    return parser

def main():
    parser = get_parser()
    args = parser.parse_args()
    template_main(
        args.shard_config_file,
        args.template_file,
        args.out_file,
        parse_vars(args.template_vars))

if __name__ == '__main__':
    main()

