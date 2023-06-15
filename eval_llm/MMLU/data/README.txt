This file contains the dev, val, and test data for our multitask test.
The dev dataset is for few-shot learning to prime the model, and the test set the source of evaluation questions.
The auxiliary_training data could be used for fine-tuning, something important for models without few-shot capabilities. This auxiliary training data comes from other NLP multiple choice datasets such as MCTest (Richardson et al., 2013), RACE (Lai et al., 2017), ARC (Clark et al., 2018, 2016), and OBQA (Mihaylov et al., 2018).
Unless otherwise specified, the questions are in reference to human knowledge as of January 1st, 2020. In the far future, it may be useful to add to the prompt that the question is written for 2020 audiences.

--

If you find this useful in your research, please consider citing the test and also the ETHICS dataset it draws from:

@article{hendryckstest2021,
  title={Measuring Massive Multitask Language Understanding},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andy Zou and Mantas Mazeika and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}

@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Dan Hendrycks and Collin Burns and Steven Basart and Andrew Critch and Jerry Li and Dawn Song and Jacob Steinhardt},
  journal={Proceedings of the International Conference on Learning Representations (ICLR)},
  year={2021}
}
