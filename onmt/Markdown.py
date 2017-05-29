# Copyright 2016 The Chromium Authors. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
import argparse


class MarkdownHelpFormatter(argparse.HelpFormatter):
    """A really bare-bones argparse help formatter that generates valid markdown.
    This will generate something like:
    usage
    # **section heading**:
    ## **--argument-one**
    ```
    argument-one help text
    ```
    """

    def _format_usage(self, usage, actions, groups, prefix):
        usage_text = super(MarkdownHelpFormatter, self)._format_usage(
                usage, actions, groups, prefix)
        return '\n```\n%s\n```\n\n' % usage_text

    def format_help(self):
        self._root_section.heading = '# %s' % self._prog
        return super(MarkdownHelpFormatter, self).format_help()

    def start_section(self, heading):
        super(MarkdownHelpFormatter, self).start_section('## **%s**' % heading)

    def _format_action(self, action):
        lines = []
        action_header = self._format_action_invocation(action)
        lines.append('### **%s** ' % action_header)
        if action.help:
            lines.append('')
            lines.append('```')
            help_text = self._expand_help(action)
            lines.extend(self._split_lines(help_text, 80))
            lines.append('```')
        lines.extend(['', ''])
        return '\n'.join(lines)


class MarkdownHelpAction(argparse.Action):
    def __init__(self, option_strings,
                 dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 **kwargs):
        super(MarkdownHelpAction, self).__init__(
                option_strings=option_strings,
                dest=dest,
                default=default,
                nargs=0,
                **kwargs)

    def __call__(self, parser, namespace, values, option_string=None):
        parser.formatter_class = MarkdownHelpFormatter
        parser.print_help()
        parser.exit()


def add_md_help_argument(parser):
    parser.add_argument('-md', action=MarkdownHelpAction,
                        help='print Markdown-formatted help text and exit.')
