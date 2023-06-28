import csv
import pandas as pd
import os

from onmt.utils.logging import init_logger
from onmt.translate.translator import build_translator
from onmt.inputters.dynamic_iterator import build_dynamic_dataset_iter
from onmt.inputters.inputter import IterOnDevice
from onmt.transforms import get_transforms_cls
from onmt.constants import CorpusTask
import onmt.opts as opts
from onmt.utils.parse import ArgumentParser
from onmt.utils.misc import use_gpu, set_random_seed

TASKS = [
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
]

data_dir = "eval_llm/MMLU/data/"


def translate_fr_mmlu_benchmark(opt):
    ArgumentParser.validate_translate_opts(opt)
    ArgumentParser._get_all_transform_translate(opt)
    ArgumentParser._validate_transforms_opts(opt)
    ArgumentParser.validate_translate_opts_dynamic(opt)
    logger = init_logger(opt.log_file)
    set_random_seed(opt.seed, use_gpu(opt))

    # Build the translator (along with the model)
    translator = build_translator(opt, logger=logger, report_score=True)

    # Build the transforms (along with the tokenizer)
    transforms_cls = get_transforms_cls(opt._all_transform)

    def translate_mmlu_tasks(subset="test"):
        for task in TASKS:
            print("Extracting %s ..." % task)
            out_file = open(
                os.path.join(data_dir, subset, task + f"_{subset}.fr.csv"),
                "w",
                encoding="utf-8",
            )
            writer = csv.writer(out_file)
            df = pd.read_csv(
                os.path.join(data_dir, subset, task + f"_{subset}.csv"), header=None
            )

            target = []
            for i in range(df.shape[0]):
                # get prompt and make sure it fits
                row = []
                for j in range(6):
                    row.append(df.iloc[i, j])
                infer_iter = build_dynamic_dataset_iter(
                    opt,
                    transforms_cls,
                    translator.vocabs,
                    task=CorpusTask.INFER,
                    src=row,
                )
                infer_iter = IterOnDevice(infer_iter, opt.gpu)
                _, preds = translator._translate(
                    infer_iter,
                    transform=infer_iter.transform,
                    attn_debug=opt.attn_debug,
                    align_debug=opt.align_debug,
                )
                translated_row = [x.lstrip() for sublist in preds for x in sublist]
                label, translated_label = row[5], translated_row[5]
                if translated_label != label:
                    print(i, row, translated_label)
                    translated_row[5] = label
                target.append(translated_row)
            writer.writerows(target)

    print("Translating dev ....")
    translate_mmlu_tasks("dev")
    print("Translating test ....")
    translate_mmlu_tasks("test")


def _get_parser():
    parser = ArgumentParser(description="run_mmlu_opennmt.py")

    opts.config_opts(parser)
    opts.translate_opts(parser, dynamic=True)
    return parser


def main():
    parser = _get_parser()
    opt = parser.parse_args()
    translate_fr_mmlu_benchmark(opt)


if __name__ == "__main__":
    main()
