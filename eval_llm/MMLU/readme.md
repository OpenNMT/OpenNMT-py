# Evaluation with MMLU dataset

All evaluations below have been computed with the OpenNMT-py converted models.

The evaluation script is tkane from the https://github.com/FranxYao/chain-of-thought-hub repo and modified to use the OpenNMT-py models

* Llama7B score (35.25) matches both the Llama paper and the score reported by chain-of-thought-hub

* Overall Llama out performs all other by a large margin. Openllama seems above the 3 others (Falcon, Redpajama, MPT)

* There are major discrepancies between tasks



| --------------------------------------- | ---------- | --------------- | -------------- | ------------ | ----------- |
|                                         | **MPT7B**  | **Redpajama7B** | **Open Llama** | **Falcon7B** | **Llama7B** |
| --------------------------------------- | ---------- | --------------- | -------------- | ------------ | ----------- |
| **ACC-all**                             | **0.2765** | **0.2745**      | **0.3007**     | **0.2765**   | **0.3525**  |
|                                         | **<br>**   | **<br>**        | **<br>**       | **<br>**     | **<br>**    |
| ACC-abstract_algebra                    | 0.2400     | 0.2500          | 0.3000         | 0.2400       | 0.2500      |
| ACC-anatomy                             | 0.2444     | 0.2667          | 0.3333         | 0.2444       | 0.3852      |
| ACC-astronomy                           | 0.2434     | 0.2763          | 0.2500         | 0.2434       | 0.3487      |
| ACC-business_ethics                     | 0.1900     | 0.2900          | 0.3200         | 0.1900       | 0.4100      |
| ACC-clinical_knowledge                  | 0.3019     | 0.3208          | 0.3887         | 0.3019       | 0.3585      |
| ACC-college_biology                     | 0.2153     | 0.3125          | 0.3264         | 0.2153       | 0.3819      |
| ACC-college_chemistry                   | 0.2300     | 0.2700          | 0.2400         | 0.2300       | 0.2900      |
| ACC-college_computer_science            | 0.3000     | 0.3100          | 0.3100         | 0.3000       | 0.2900      |
| ACC-college_mathematics                 | 0.2900     | 0.2500          | 0.2800         | 0.2900       | 0.3400      |
| ACC-college_medicine                    | 0.2659     | 0.2659          | 0.3179         | 0.2659       | 0.3237      |
| ACC-college_physics                     | 0.2157     | 0.2451          | 0.1863         | 0.2157       | 0.2451      |
| ACC-computer_security                   | 0.2800     | 0.3600          | 0.3800         | 0.2800       | 0.4500      |
| ACC-conceptual_physics                  | 0.3149     | 0.2723          | 0.3064         | 0.3149       | 0.3702      |
| ACC-econometrics                        | 0.2632     | 0.2368          | 0.2895         | 0.2632       | 0.2632      |
| ACC-electrical_engineering              | 0.2828     | 0.3034          | 0.3034         | 0.2828       | 0.2483      |
| ACC-elementary_mathematics              | 0.2593     | 0.2646          | 0.2698         | 0.2593       | 0.2646      |
| ACC-formal_logic                        | 0.1905     | 0.4048          | 0.2381         | 0.1905       | 0.2619      |
| ACC-global_facts                        | 0.3100     | 0.3200          | 0.3200         | 0.3100       | 0.3000      |
| ACC-high_school_biology                 | 0.2645     | 0.2484          | 0.2968         | 0.2645       | 0.3387      |
| ACC-high_school_chemistry               | 0.2512     | 0.2660          | 0.2512         | 0.2512       | 0.2956      |
| ACC-high_school_computer_science        | 0.3200     | 0.2700          | 0.2800         | 0.3200       | 0.3300      |
| ACC-high_school_european_history        | 0.2909     | 0.2848          | 0.3455         | 0.2909       | 0.4667      |
| ACC-high_school_geography               | 0.1667     | 0.3283          | 0.3333         | 0.1667       | 0.3333      |
| ACC-high_school_government_and_politics | 0.2591     | 0.2124          | 0.3575         | 0.2591       | 0.4611      |
| ACC-high_school_macroeconomics          | 0.2615     | 0.2718          | 0.3564         | 0.2615       | 0.3410      |
| ACC-high_school_mathematics             | 0.2481     | 0.2667          | 0.2407         | 0.2481       | 0.2630      |
| ACC-high_school_microeconomics          | 0.2899     | 0.3067          | 0.2941         | 0.2899       | 0.3319      |
| ACC-high_school_physics                 | 0.3179     | 0.2649          | 0.2517         | 0.3179       | 0.2649      |
| ACC-high_school_psychology              | 0.2440     | 0.3229          | 0.3505         | 0.2440       | 0.4789      |
| ACC-high_school_statistics              | 0.1852     | 0.2454          | 0.3981         | 0.1852       | 0.3241      |
| ACC-high_school_us_history              | 0.2892     | 0.2255          | 0.3137         | 0.2892       | 0.3284      |
| ACC-high_school_world_history           | 0.2996     | 0.2785          | 0.2869         | 0.2996       | 0.4262      |
| ACC-human_aging                         | 0.4215     | 0.1659          | 0.2870         | 0.4215       | 0.3991      |
| ACC-human_sexuality                     | 0.2901     | 0.2519          | 0.2748         | 0.2901       | 0.3435      |
| ACC-international_law                   | 0.2479     | 0.2231          | 0.3636         | 0.2479       | 0.5207      |
| ACC-jurisprudence                       | 0.3426     | 0.2315          | 0.3426         | 0.3426       | 0.4167      |
| ACC-logical_fallacies                   | 0.2638     | 0.2638          | 0.2883         | 0.2638       | 0.4172      |
| ACC-machine_learning                    | 0.3750     | 0.2232          | 0.2321         | 0.3750       | 0.2768      |
| ACC-management                          | 0.2816     | 0.2816          | 0.2524         | 0.2816       | 0.3301      |
| ACC-marketing                           | 0.2949     | 0.2735          | 0.3761         | 0.2949       | 0.4615      |
| ACC-medical_genetics                    | 0.2800     | 0.2400          | 0.2700         | 0.2800       | 0.3700      |
| ACC-miscellaneous                       | 0.2976     | 0.2899          | 0.3678         | 0.2976       | 0.4278      |
| ACC-moral_disputes                      | 0.3092     | 0.2659          | 0.3295         | 0.3092       | 0.4133      |
| ACC-moral_scenarios                     | 0.2492     | 0.2469          | 0.2469         | 0.2492       | 0.2425      |
| ACC-nutrition                           | 0.2582     | 0.2908          | 0.3301         | 0.2582       | 0.3922      |
| ACC-philosophy                          | 0.2830     | 0.2830          | 0.2830         | 0.2830       | 0.4051      |
| ACC-prehistory                          | 0.3117     | 0.3210          | 0.3210         | 0.3117       | 0.3519      |
| ACC-professional_accounting             | 0.2979     | 0.2872          | 0.2553         | 0.2979       | 0.2730      |
| ACC-professional_law                    | 0.2497     | 0.2705          | 0.2523         | 0.2497       | 0.2973      |
| ACC-professional_medicine               | 0.3125     | 0.2059          | 0.2500         | 0.3125       | 0.4265      |
| ACC-professional_psychology             | 0.2647     | 0.2925          | 0.2696         | 0.2647       | 0.3546      |
| ACC-public_relations                    | 0.3364     | 0.3182          | 0.4091         | 0.3364       | 0.4091      |
| ACC-security_studies                    | 0.3102     | 0.2816          | 0.2939         | 0.3102       | 0.3306      |
| ACC-sociology                           | 0.3532     | 0.2587          | 0.2488         | 0.3532       | 0.4726      |
| ACC-us_foreign_policy                   | 0.4200     | 0.3200          | 0.3900         | 0.4200       | 0.4300      |
| ACC-virology                            | 0.3554     | 0.2530          | 0.3494         | 0.3554       | 0.3253      |
| ACC-world_religions                     | 0.3333     | 0.3041          | 0.4035         | 0.3333       | 0.4912      |
