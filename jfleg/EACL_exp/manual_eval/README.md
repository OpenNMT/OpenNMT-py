# Description of Manual Evaluation

100 sentences from the JFLEG dev set were randomly chosen and the authors examined 6 versions of each sentence:

  - the original sentence,
  - each system's correction
  - a randomly selected reference correction

We annotated each version of the sentences with the following codes:

- Error types:
  - **A**: Sounds awkward (not fluent)
  - **O**: Has an orthographic error (spelling or capitalization)
  - **U**: Ungrammatical

- Edit types:
  - **F**: Contains a fluency edit
  - **M**: Contains a minimal edit

- Other:
  - **S**: Original sentence should be split into multiple sentences.

The annotations are found in `EACL_exp/manual_eval/coded_sentences.csv`.
