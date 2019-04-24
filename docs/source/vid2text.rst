This tutorial shows how to replicate the results from
`"Describing Videos by Exploiting Temporal Structure" <https://arxiv.org/pdf/1502.08029.pdf>`_
[`code <https://github.com/yaoli/arctic-capgen-vid>`_]
using OpenNMT-py.

Get `YouTubeClips.tar` from `here <http://www.cs.utexas.edu/users/ml/clamp/videoDescription/>`_.
Use ``tar -xvf YouTubeClips.tar`` to decompress the archive.

Now, visit `this repo <https://github.com/yaoli/arctic-capgen-vid>`_.
Follow the "preprocessed YouTube2Text download link."
We'll be throwing away the Googlenet features. We just need the captions.
Use ``unzip youtube2text_iccv15.zip`` to decompress the files.

Get to the following directory structure: ::

    yt2t
    |-YouTubeClips
    |-youtube2text_iccv15

Change directories to `yt2t`. We'll rename the videos to follow the "vid#.avi" format:

.. code-block:: python

    import pickle
    import os


    YT = "youtube2text_iccv15"
    YTC = "YouTubeClips"

    # load the YouTube hash -> vid### map.
    with open(os.path.join(YT, "dict_youtube_mapping.pkl"), "rb") as f:
        yt2vid = pickle.load(f, encoding="latin-1")

    for f in os.listdir(YTC):
        hashy, ext = os.path.splitext(f)
        vid = yt2vid[hashy]
        fpath_old = os.path.join(YTC, f)
        f_new = vid + ext
        fpath_new = os.path.join(YTC, f_new)
        os.rename(fpath_old, fpath_new)

Make sure all the videos have the same (low) framerate using

.. code-block:: bash

    for fi in $( ls ); do ffmpeg -y -i $fi -r 2 $fi; done

Now we want to convert the frames into sequences of CNN feature vectors.
(We'll use the environment variable ``Y2T2`` to refer to the `yt2t` directory.)

.. code-block:: bash

    export YT2T=`pwd`

Then change directories back to the `OpenNMT-py` directory.
Use `tools/img_feature_extractor.py`.
Set the ``--world_size`` argument to the number of GPUs you have available
(You can use the environment variable ``CUDA_VISIBLE_DEVICES`` to restrict the GPUs used).

.. code-block:: bash

    PYTHONPATH=$PWD:$PYTHONPATH python tools/vid_feature_extractor.py --root_dir $YT2T/YouTubeClips --out_dir $YT2T/r152

Ensure the count is equal to 1970.
You can use ``ls -1 $YT2T/r152 | wc -l``.
If not, rerun the script. It will only process on the missing feature vectors.
(Note this is unexpected behavior and consider opening an issue.)

Now we turn our attention to the annotations. Each video has multiple associated captions. We want to
train the model on each video + single caption pair. We'll collect all the captions per video, then we'll
flatten them into files listing the feature vector sequence filenames (repeating for each caption) and the
annotations. We skip the test videos since they are handled separately at translation time.

Change directories back to ``YT2T``:

.. code-block:: bash

    cd $YT2T

.. code-block:: python

    import pickle
    import os
    from random import shuffle


    YT = "youtube2text_iccv15"
    SHUFFLE = True

    with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
        ann = pickle.load(f, encoding="latin-1")

    vid2anns = {}
    for vid_name, data in ann.items():
        for d in data:
            try:
                vid2anns[vid_name].append(d["tokenized"])
            except KeyError:
                vid2anns[vid_name] = [d["tokenized"]]

    with open(os.path.join(YT, "train.pkl"), "rb") as f:
        train = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "valid.pkl"), "rb") as f:
        val = pickle.load(f, encoding="latin-1")

    with open(os.path.join(YT, "test.pkl"), "rb") as f:
        test = pickle.load(f, encoding="latin-1")

    train_files = open("yt2t_train_files.txt", "w")
    val_files = open("yt2t_val_files.txt", "w")
    test_files = open("yt2t_test_files.txt", "w")

    train_cap = open("yt2t_train_cap.txt", "w")
    val_cap = open("yt2t_val_cap.txt", "w")

    vid_names = vid2anns.keys()
    if SHUFFLE:
        vid_names = list(vid_names)
        shuffle(vid_names)


    for vid_name in vid_names:
        anns = vid2anns[vid_name]
        vid_path = vid_name + ".npy"
        for i, an in enumerate(anns):
            an = an.replace("\n", " ")  # some caps have newlines
            split_name = vid_name + "_" + str(i)
            if split_name in train:
                train_files.write(vid_path + "\n")
                train_cap.write(an + "\n")
            elif split_name in val:
                val_files.write(vid_path + "\n")
                val_cap.write(an + "\n")
            else:
                # Don't need to save out the test captions,
                # just the files. And, don't need to repeat
                # it for each caption
                assert split_name in test
                if i == 0:
                    test_files.write(vid_path + "\n")

Return to the `OpenNMT-py` directory. Now we preprocess the data for training.
We preprocess with a small shard size of 1000. This keeps the amount of data in memory (RAM) to a
manageable 10 G. If you have more RAM, you can increase the shard size.

Preprocess the data with

.. code-block:: bash

    python preprocess.py -data_type vec -train_src $YT2T/yt2t_train_files.txt -src_dir $YT2T/r152/ -train_tgt $YT2T/yt2t_train_cap.txt -valid_src $YT2T/yt2t_val_files.txt -valid_tgt $YT2T/yt2t_val_cap.txt -save_data data/yt2t --shard_size 1000

Train with

.. code-block:: bash

    python train.py -data data/yt2t -save_model yt2t-model -world_size 2 -gpu_ranks 0 1 -model_type vec -batch_size 64 -train_steps 10000 -valid_steps 500 -save_checkpoint_steps 500 -encoder_type brnn -optim adam -learning_rate .0001 -feat_vec_size 2048

Translate with

.. code-block::

    python translate.py -model yt2t-model_step_7200.pt -src $YT2T/yt2t_test_files.txt -output pred.txt -verbose -data_type vec -src_dir $YT2T/r152 -gpu 0 -batch_size 10

.. note::

    Generally, you want to keep the model that has the lowest validation perplexity. That turned out to be
    at step 7200, but choosing a different validation frequency or random seed could result in different results.


Then you can use `coco-caption <https://github.com/tylin/coco-caption/tree/master/pycocoevalcap>`_ to evaluate the predictions.
(Note that the fork `flauted <https://github.com/flauted/coco-caption>`_ can be used for Python 3 compatibility).
Install the git repository with pip using


.. code-block:: bash

    pip install git+<clone URL>

Then use the following Python code to evaluate:

.. code-block:: python

    import os
    from pprint import pprint
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.spice.spice import Spice


    if __name__ == "__main__":
        pred = open("pred.txt")

        import pickle
        import os

        YT = os.path.join(os.environ["YT2T"], "youtube2text_iccv15")

        with open(os.path.join(YT, "CAP.pkl"), "rb") as f:
            ann = pickle.load(f, encoding="latin-1")

        vid2anns = {}
        for vid_name, data in ann.items():
            for d in data:
                try:
                    vid2anns[vid_name].append(d["tokenized"])
                except KeyError:
                    vid2anns[vid_name] = [d["tokenized"]]

        test_files = open(os.path.join(os.environ["YT2T"], "yt2t_test_files.txt"))

        scorers = {
            "Bleu": Bleu(4),
            "Meteor": Meteor(),
            "Rouge": Rouge(),
            "Cider": Cider(),
            "Spice": Spice()
        }

        gts = {}
        res = {}
        for outp, filename in zip(pred, test_files):
            filename = filename.strip("\n")
            outp = outp.strip("\n")
            vid_id = os.path.splitext(filename)[0]
            anns = vid2anns[vid_id]
            gts[vid_id] = anns
            res[vid_id] = [outp]

        scores = {}
        for name, scorer in scorers.items():
            score, all_scores = scorer.compute_score(gts, res)
            if isinstance(score, list):
                for i, sc in enumerate(score, 1):
                    scores[name + str(i)] = sc
            else:
                scores[name] = score
        pprint(scores)

Here are our results ::

    {'Bleu1': 0.7888553878084233,
     'Bleu2': 0.6729376621109295,
     'Bleu3': 0.5778428507344473,
     'Bleu4': 0.47633625833397897,
     'Cider': 0.7122415518428051,
     'Meteor': 0.31829562714082704,
     'Rouge': 0.6811305229481235,
     'Spice': 0.044147089472463576}


So how does this stack up against the paper? These results should be compared to the "Global (Temporal Attention)"
row in Table 1. The authors report BLEU4 0.4028, METEOR 0.2900, and CIDEr 0.4801. So, our results are a significant
improvement. Our architecture follows the general encoder + attentional decoder described in the paper, but the
actual attention implementation is slightly different. The paper downsamples by choosing 26 equally spaced frames from
the first 240, while we downsample the video to 2 fps. Also, we use ResNet features instead of GoogLeNet, and we
lowercase while the paper does not, so some improvement is expected.
