# Neural AMR

[Torch](http://torch.ch) implementation of sequence-to-sequence models for AMR parsing and generation based on the [Harvard NLP](https://github.com/sinantie/NeuralAmr/edit/master/README.md) framework. We provide the code for pre-processing, anonymizing, de-anonymizing, training and predicting from and to AMR. We are also including pre-trained models on 20M sentences from Gigaword and fine-tuned on the AMR [LDC2015E86: DEFT Phase 2 AMR Annotation R1 Corpus](https://catalog.ldc.upenn.edu/LDC2015E86). You can find all the details in the following paper:

- [Neural AMR: Sequence-to-Sequence Models for Parsing and Generation](https://arxiv.org/abs/1704.08381). (Ioannis Konstas, Srinivasan Iyer, Mark Yatskar, Yejin Choi, Luke Zettlemoyer. ACL 2017)

## Requirements

The pre-trained models only run on *GPUs*, so you will need to have the following installed:

- Latest [NVIDIA driver](http://www.nvidia.com/Download/index.aspx)
- [CUDA 8 Toolkit](https://developer.nvidia.com/cuda-toolkit)
- [cuDNN](https://developer.nvidia.com/cudnn) (The NVIDIA CUDA Deep Neural Network library)
- [Torch](http://torch.ch/docs/getting-started.html)

## Installation

- Install the following packages for Torch using `luarocks`:
```
nn nngraph cutorch cunn cudnn
```
- Install the Deepmind version of `torch-hdf5` from [here](https://github.com/deepmind/torch-hdf5/blob/master/doc/usage.md).

*(Only for training models)* 

- Install `cudnn.torch` from [here](https://github.com/soumith/cudnn.torch).
- Install the following packages for Python 2.7: 
```
pip install numpy h5py
```

*(Only for downloading the pretrained models)*

- Download and unzip the models from [here](https://drive.google.com/file/d/0B0e2gHbz7CcIc0p1SjRhLVR2bTA/view?usp=sharing)

- Export the cuDNN library path (you can add it to your .bashrc or .profile):
```
export CUDNN_PATH="path_to_cudnn/lib64/libcudnn.so"
```

- Or instead of the previous step you can copy the cuDNN library files into /usr/local/cuda/lib64/ or to the corresponding folders in the CUDA directory.

## Usage 

### AMR Generation
You can generate text from AMR graphs using our pre-trained model on 20M sentences from Gigaword, in two different ways:
- By running an interactive tool that reads input from `stdin`:
```
./generate_amr_single.sh [stripped|full|anonymized]
```

- By running the prediction on a single file, which contains an AMR graph per line:
```
./generate_amr.sh input_file [stripped|full|anonymized]
```

You can optionally provide an argument that tells the system to accept either `full` AMR as described in the [annotation guidelines](https://github.com/amrisi/amr-guidelines/blob/master/amr.md), or a `stripped` version, which removes variables, senses, parentheses from leaves, and assumes a simpler markup for Named Entities, date mentions, and numbers. You can also provide the input in `anonymized` format, i.e., similar to `stripped` but with Named Entities, date mentions, and numbers anonymized.

An example using the `full` format:
```
(h / hold-04 :ARG0 (p2 / person :ARG0-of (h2 / have-org-role-91 :ARG1 (c2 / country :name (n3 / name :op1 "United" :op2 "States")) :ARG2 (o / official)))  :ARG1 (m / meet-03 :ARG0 (p / person  :ARG1-of (e / expert-01) :ARG2-of (g / group-01))) :time (d2 / date-entity :year 2002 :month 1) :location (c / city  :name (n / name :op1 "New" :op2 "York")))
```

The same example using the `stripped` format:
```
hold :ARG0 ( person :ARG0-of ( have-org-role :ARG1 (country :name "United States") :ARG2 official)) :ARG1 (meet :ARG0 (person  :ARG1-of expert :ARG2-of  group)) :time (date-entity :year 2002 :month 1) :location (city :name "New York" )
```

The same example using the `anonymized` format:
```
hold :ARG0 ( person :ARG0-of ( have-org-role :ARG1 location_name_0 :ARG2 official ) ) :ARG1 ( meet :ARG0 ( person :ARG1-of expert :ARG2-of group ) ) :time ( date-entity year_date-entity_0 month_date-entity_0 ) :location location_name_1
```

For full details and more examples, see [here](). 

### AMR Parsing

You can also parse text to the corresponding AMR graph, using our pre-trained model on 20M sentences from Gigaword.

Similarly to AMR generation, you can parse text in two ways:

- By running an interactive tool that reads text from `stdin`:
```
./parse_amr_single.sh [text|textAnonymized]
```

- By running the prediction on a single file, which contains a sentence per line:
```
./parse_amr.sh input_file [text|textAnonymized]
```

You can optionally provide an argument to the scripts that inform them to either accept `text` and perform NE recognition and anonymization on it, or bypass this process entirely (`textAnonymized`).

### Script Options (generate_amr.sh, generate_amr_single.sh, parse_amr.sh, parse_amr_single.sh)
- `interactive_mode [0,1]`: Set `0` for generating from a file, or `1` to generate from `stdin`.
- `model [str]`: The path to the trained model.
- `input_type [stripped|full]` (**AMR Generation only**): Set `full` for standard AMR graph input, or `stripped` which expects AMR graphs with no variables, senses, parentheses from leaves, and assumes a simpler markup for Named Entities (for more details and examples, see [here]()).
- `src_file [str]`: The path to the input file that contains AMR graphs, one per line.
- `gpuid [int]`: The GPU id number.
- `src_dict, targ_dict [str]`: Path to source and target dictionaries. These are usually generated during preprocessing of the corpus. ==Note==: `src_dict` and `targ_dict` paths need to be reversed when generating text or parsing to AMR.
- `beam [int]`: The beam size of the decoder (default is 5).
- `replace_unk [0,1]`: Replace unknown words with either the input token that has the highest attention weight, or the word that maps to the input token as provided in `srctarg_dict`.
- `srctarg_dict [str]`: Path to source-target dictionary to replace unknown tokens. Each line should be a source token and its corresponding target token, separated by `|||` (see `resources/training-amr-nl-alignments.txt`).
- `max_sent_l [str]`: Maximum sentence length (default is 507, i.e., the longest input AMR graph or sentence (depending on the task) in number of tokens from the dev set). If any of the sequences in `src_file` are longer than this it will error out.
 

### (De-)Anonymization Process
The source code for the whole anonymization/deanonymization pipeline is provided under the `java/AmrUtils` folder. You can rebuild the code by running the script:

```
./rebuild_AmrUtils.sh
```

This should create the executable `lib/AmrUtils.jar`.
The (de-)anonymization tools are generally controlled using the following shell script command (==Note== that it is automatically being called inside the lua code when parsing/generating, so generally you don't need to deal with it when running the scripts described above). The first argument denotes the specific (de-)anonymization to perform, the second argument specifies whether the input comes either from stdin or from a file, where each input is provided one per line:

```
./anonDeAnon_java.sh anonymizeAmrStripped|anonymizeAmrFull|deAnonymizeAmr|anonymizeText|deAnonymizeText input_isFile[true|false] input
```

- ==Note==: In order to anonymize text sentences, you need to run the Stanford NER server first (you can just execute it in the background):
   ```
   ./nerServer.sh&
   ```

   Optionally you can provide a port number as an argument.


There are four main operations you can perform with the tools, namely anonymization of AMR graphs, anonymization of text sentences, deAnonymization of (predicted) sentences, and deAnonymization of (predicted) AMR graphs.:

- Anonymize an AMR graph (`anonymizeAmrStripped, anonymizeAmrFull`)
In this case, you provide an input representing a stripped or full AMR graph, and the script outputs the **anonymized graph** (in the case of full it also strips it down, i.e., removes variable names, instance-of relations, most brackets, and simplifies NEs/dates/number subgraphs of the input), the **anonymization alignments** (useful for deAnonymizing the corresponding predicted sentence later), and the nodes/edges of the graph in an un-ordered JSON format (useful for visualization tools such as [vis.js](http://visjs.org/)). The three outputs are delimited using the special character `#`. For example:
   ```
   ./anonDeAnon_java.sh anonymizeAmrFull false "(h / hello :arg1 (p / person :name (n / name :op1 \"John\" :op2 \"Doe\")))"
   ```
   should give the output:
   ```
   hello :arg1 person_name_0#person_name_0|||name_John_Doe#
   "nodes":[{"id":1,"label":"hello"},{"id":2,"label":"person"},{"id":3,"label":"name"},{"id":4,"label":"\"John\""},{"id":5,"label":"\"Doe\""}],
   "edges":[{"from":1,"to":2,"label":"arg1"},{"from":2,"to":3,"label":"name"},{"from":3,"to":4,"label":"op1"},{"from":3,"to":5,"label":"op2"}]
   ```
   
   Anonymization alignments have the format:
   ```
   amr-anonymized-token|||type_concatenated-AMR-tokens
   ```    
   Finally, multiple anonymization alignments for the same sentence, are tab-delimeted.

- Anonymize a text sentence (`anonymizeText`)
   Remember that you need to have the NER server running, as explained above. In this example you simply provide the sentence as in input. For example:
   ```
   ./anonDeAnon_java.sh anonymizeText false "My name is John Doe"
   ```
   should give the output:
   ```
   my name is person_name_0#person_name_0|||John Doe
   ```
   Note that the anonymization alignments from text are slightly different than the ones from AMR graphs; the second part is a span of the text separated with space.

- De-anonymize an AMR graph (`deAnonymizeAmr`)
   In this case, you provide an input representing a stripped AMR graph, as well as the corresonding anonymization alignments provided from a previous run of the script using the ==anonymizeText== option, **delimited** by `#`, and the script outputs the de-anonymized AMR graph, as well as the nodes/edges of the graph in an un-ordered JSON format  (useful for visualization tools such as [vis.js](http://visjs.org/)). For example:
   ```
   ./anonDeAnon_java.sh deAnonymizeAmr false "hello :arg1 person_name_0#person_name_0|||John Doe"
   ```
   should give the output:
   ```
   (h / hello :arg1 (p / person :name (n / name :op1 "John" :op2 "Doe")))#
   "nodes":[{"id":1,"label":"hello"},{"id":2,"label":"person"},{"id":3,"label":"name"},{"id":4,"label":"\"John\""},{"id":5,"label":"\"Doe\""}],
   "edges":[{"from":1,"to":2,"label":"arg1"},{"from":2,"to":3,"label":"name"},{"from":3,"to":4,"label":"op1"},{"from":3,"to":5,"label":"op2"}]
   ```

- De-anonymize a text sentence (`deAnonymizeText`)
  Simply, provide the anonymized (predicted) text sentence, along with the anonymization alingmnets produced from a previous run of the tool using the ==anonymizeAmrFull/Stripped== option, **delimited** by `#`. The script should output the de-anonymized text sentence. For example:
   ```
   ./anonDeAnon_java.sh deAnonymizeText false "my name is person_name_0#person_name_0|||name_John_Doe"
   ```
   should give the output:
   ```
   my name is John Doe
   ```

Finally, when running the tool with the input being in a file (provide the path as the 3rd argument of the script, and set the 2nd argument to true), you always need to provide the **original** files containing the AMR graphs/sentences only. The tool will then automatically create the corresponding anonymized file (`*.anonymized`), as well as the anonymization alignments' file (`*.alignments`) during anonymization. Similarly, when de-anonymizing it will automatically look for the (`*.anonymized`, and `*.alignments`) files and create a new resulting file with the extension (`*.pred`).

#### (De)-Anonymizing Parallel Corpus (e.g., LDC versions)

If you have a parallel corpus, such as the LDC2015E86 that was used to train the models in this work, or [Little Prince](https://amr.isi.edu/download.html), which is included in this repository as well for convenience, then you need to follow a slightly different procedure. 

The idea is to use alignments between the AMR graphs and corresponding text, in order to accurately identify the entities that will get anonymized. The alignments can be either obtained using the [unsupervised aligner](https://www.isi.edu/~damghani/papers/Aligner.zip) by Nima Pourdamghani, or [JAMR](https://github.com/jflanigan/jamr) by Jeff Flanigan. If you are using the annotated LDC versions, then they should already be automatically aligned using the first aligner (use files under the folder `alignments/`). The code in this repository supports either or both types of alignments. 

1. In order to get alignments from JAMR on the provided Little Prince corpus do the following:

- First, concatenate the training, dev and test splits to a single file (e.g., `little-prince-all.txt`) for convenience.
- Change to the path you downloaded and installed JAMR, and execute the command:
```
scripts/ALIGN.sh path/to/NeuralAmr/resources/sample-data/little-prince/little-prince-all.txt > /path/to/NeuralAmr/resources/sample-data/little-prince/jamr-alignments-all.txt
```

2. Open `settings.properties` and make sure `amr.down.base` and `amr.jamr.alignments` point to the right folders (they point to Little Prince directory by default). 
You can also enable or disable some pre-processing functionalities from here as well, such as whether to use Named Entity clusters (person, organization, location, and other instead of the fine-grained AMR categories; you can alter the clusters through the property `amr.concepts.ne`) via the property `amr.down.useNeClusters` (default is `true` for preparing the corpus for Generation, and should be set to `false` for Parsing). Similarly, you might want to enable output of senses on concepts (e.g., `say-01`, instead of just `say`) via the property `amr.down.outputSense` (default is `false` for Generation and `true` for Parsing).
Another important property is `amr.down.input` which specifies which portion of the corpus to process (default is `training,dev,test` which are the folder names in the LDC corpora and Little Prince).

3. Preprocess and anonymize the corpus by executing the script:
```
./anonParallel_java.sh
``` 
This will take care of the proper bracket pre-processing anonymization, and splitting of the corpus to training, dev and test source and target files that can be directly used for training and evaluating your models.
There are three options to change there:
- `DATA_DIR` which points to a directory that will hold pre-processed files with many interesting meta-data, such as vocabularies, alignments, anonymization pairs, histograms and so on.
- `OUT_DIR` which refers to a directory which contains only the essential anonymized training, dev, test source and target files, as well as the anonymization alignments in separate files. 
- `suffix` which is a handy parameter for changing the name of `OUT_DIR` directory.

4. (Generation only) De-anonymize and automatically evaluate the output of a model using averaged BLEU, METEOR and multiBLEU by executing the script:
```
./recomputeMetrics.sh [INPUT_PATH REF_PATH]
```
The script contains three important options:
- `DATASET` which refers to the portion of the set to evaluate against; default is `dev` (the other option is `test`).
- `DATA_PATHNAME` which points to the preprocessed corpus directory created from the previous script, which contains the reference data. Normally, it should be the same as `OUT_DIR` from above.
- `INPUT_PATH` which is the folder containing the file(s) with anonymized predictions. In case there are multiple files, for example from different epoch runs, then the code automatically processes all of them and reports back the one with the highest multiBLEU score.
