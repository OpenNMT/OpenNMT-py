# Evaluation with MMLU-FR dataset

All evaluations below have been computed with the OpenNMT-py converted models.

The evaluation script is tkane from the https://github.com/FranxYao/chain-of-thought-hub repo and modified to use the OpenNMT-py models



|                                         | **Llama 7B** | **Llama 13B** |
| --------------------------------------- | ------------ | ------------- |
| **ACC-all**                             | **0.3193**   | **0.3836**    |
|                                         |              |               |
| ACC-abstract_algebra                    | 0.2800       | 0.3200        |
| ACC-anatomy                             | 0.3630       | 0.3704        |
| ACC-astronomy                           | 0.2829       | 0.3355        |
| ACC-business_ethics                     | 0.3500       | 0.3800        |
| ACC-clinical_knowledge                  | 0.2981       | 0.4038        |
| ACC-college_biology                     | 0.3403       | 0.3889        |
| ACC-college_chemistry                   | 0.3600       | 0.2200        |
| ACC-college_computer_science            | 0.2900       | 0.3700        |
| ACC-college_mathematics                 | 0.2800       | 0.3800        |
| ACC-college_medicine                    | 0.2659       | 0.3757        |
| ACC-college_physics                     | 0.1471       | 0.2745        |
| ACC-computer_security                   | 0.3900       | 0.4400        |
| ACC-conceptual_physics                  | 0.2851       | 0.3745        |
| ACC-econometrics                        | 0.2456       | 0.2018        |
| ACC-electrical_engineering              | 0.3655       | 0.3172        |
| ACC-elementary_mathematics              | 0.2513       | 0.2381        |
| ACC-formal_logic                        | 0.2540       | 0.3810        |
| ACC-global_facts                        | 0.3400       | 0.3100        |
| ACC-high_school_biology                 | 0.3032       | 0.4000        |
| ACC-high_school_chemistry               | 0.2956       | 0.2365        |
| ACC-high_school_computer_science        | 0.3200       | 0.4000        |
| ACC-high_school_european_history        | 0.3818       | 0.4848        |
| ACC-high_school_geography               | 0.2879       | 0.4040        |
| ACC-high_school_government_and_politics | 0.2902       | 0.5078        |
| ACC-high_school_macroeconomics          | 0.3103       | 0.3872        |
| ACC-high_school_mathematics             | 0.2111       | 0.2519        |
| ACC-high_school_microeconomics          | 0.3025       | 0.3487        |
| ACC-high_school_physics                 | 0.3113       | 0.2781        |
| ACC-high_school_psychology              | 0.3798       | 0.4716        |
| ACC-high_school_statistics              | 0.3981       | 0.2546        |
| ACC-high_school_us_history              | 0.3284       | 0.3676        |
| ACC-high_school_world_history           | 0.3165       | 0.5865        |
| ACC-human_aging                         | 0.3632       | 0.3991        |
| ACC-human_sexuality                     | 0.3740       | 0.4962        |
| ACC-international_law                   | 0.4711       | 0.6033        |
| ACC-jurisprudence                       | 0.3889       | 0.4444        |
| ACC-logical_fallacies                   | 0.3190       | 0.3926        |
| ACC-machine_learning                    | 0.2589       | 0.2500        |
| ACC-management                          | 0.3107       | 0.5437        |
| ACC-marketing                           | 0.3974       | 0.5855        |
| ACC-medical_genetics                    | 0.3000       | 0.5300        |
| ACC-miscellaneous                       | 0.3742       | 0.5223        |
| ACC-moral_disputes                      | 0.3237       | 0.4306        |
| ACC-moral_scenarios                     | 0.2402       | 0.2492        |
| ACC-nutrition                           | 0.3366       | 0.4412        |
| ACC-philosophy                          | 0.4341       | 0.4373        |
| ACC-prehistory                          | 0.3426       | 0.4043        |
| ACC-professional_accounting             | 0.3014       | 0.2766        |
| ACC-professional_law                    | 0.2901       | 0.3110        |
| ACC-professional_medicine               | 0.4265       | 0.4338        |
| ACC-professional_psychology             | 0.2974       | 0.3513        |
| ACC-public_relations                    | 0.2545       | 0.4182        |
| ACC-security_studies                    | 0.2490       | 0.4204        |
| ACC-sociology                           | 0.4229       | 0.5373        |
| ACC-us_foreign_policy                   | 0.4100       | 0.5000        |
| ACC-virology                            | 0.3554       | 0.3494        |
| ACC-world_religions                     | 0.3977       | 0.6023        |
