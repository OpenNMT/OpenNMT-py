import argparse
import os
import time
import pyrouge
import shutil


def test_rouge(cand_file, ref_file):
    f_cand = open(cand_file, encoding="utf-8")
    f_ref = open(ref_file, encoding="utf-8")
    current_time = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime())
    tmp_dir = ".rouge-tmp-{}".format(current_time)
    if not os.path.isdir(tmp_dir):
        os.mkdir(tmp_dir)
        os.mkdir(tmp_dir + "/candidate")
        os.mkdir(tmp_dir + "/reference")
    for i, line in enumerate(f_cand):
        with open(tmp_dir + "/candidate/cand.{}.txt".format(str(i).zfill(3)), "w", encoding="utf-8") as f:
            f.write(line)
    for i, line in enumerate(f_ref):
        with open(tmp_dir + "/reference/ref.{}.txt".format(str(i).zfill(3)), "w", encoding="utf-8") as f:
            f.write(line)
    f_cand.close()
    f_ref.close()
    r = pyrouge.Rouge155()
    r.model_dir = tmp_dir + "/reference/"
    r.system_dir = tmp_dir + "/candidate/"
    r.model_filename_pattern = 'ref.#ID#.txt'
    r.system_filename_pattern = 'cand.(\d+).txt'
    rouge_results = r.convert_and_evaluate()
    results_dict = r.output_to_dict(rouge_results)

    shutil.rmtree(tmp_dir)
    return results_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', type=str, default="candidate.txt", help='candidate file')
    parser.add_argument('-r', type=str, default="reference.txt", help='reference file')
    args = parser.parse_args()
    results_dict = test_rouge(args.c, args.r)
    print(">> ROUGE(1/2/3/L/SU4): {:.2f}/{:.2f}/{:.2f}/{:.2f}/{:.2f}".format(
        results_dict["rouge_1_f_score"] * 100, results_dict["rouge_2_f_score"] * 100,
        results_dict["rouge_3_f_score"] * 100, results_dict["rouge_l_f_score"] * 100,
        results_dict["rouge_su*_f_score"] * 100))
