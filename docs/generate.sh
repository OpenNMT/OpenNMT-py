#! /bin/sh

gen_script_options ()
{
    echo "<!--- This file was automatically generated. Do not modify it manually but use the docs/options/generate.sh script instead. -->" > $2
    echo "" >> $2
    python3 $1 -md >> $2
}

gen_script_options preprocess.py docs/options/preprocess.md
gen_script_options train.py docs/options/train.md
gen_script_options translate.py docs/options/translate.md
