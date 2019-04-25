cat data/ldc2015e86_ordered/test/deft-p2-amr-r1-amrs-training-all.txt | grep '# ::id ' | cut -d' ' -f 3 > data/ldc2015e86_ordered/meta/training_original_ids.txt
cat data/ldc2015e86_ordered/dev/deft-p2-amr-r1-amrs-dev-all.txt | grep '# ::id ' | cut -d' ' -f 3 > data/ldc2015e86_ordered/meta/dev_original_ids.txt
cat data/ldc2015e86_ordered/test/deft-p2-amr-r1-amrs-test-all.txt | grep '# ::id ' | cut -d' ' -f 3 > data/ldc2015e86_ordered/meta/test_original_ids.txt
