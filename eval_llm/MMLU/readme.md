# Evaluation with MMLU dataset

All evaluations below have been computed with the OpenNMT-py converted models.

The evaluation script is tkane from the https://github.com/FranxYao/chain-of-thought-hub repo and modified to use the OpenNMT-py models

* Llama7B score (35.25) matches both the Llama paper and the score reported by chain-of-thought-hub

* Falcon7B is a little higher then the score reported by chain-of-thought-hub (0.2641)

* I ran MPT7B with chain-of-thought-hub and found 28.46, again ours is a little higher.

* There are major discrepancies between those scores and Open LLM leaderboard of HF for MPT, Falcon, Redpajama that are way higher on the leaderboard.


|                                         | **MPT7B**  | **Redpajama7B** | **Open Llama7B** | **Falcon7B** | **Llama7B** | **Open Llama13B** | **Llama13B** |
| --------------------------------------- | ---------- | --------------- | ---------------- | ------------ | ----------- | ----------------- | ------------ |
| **ACC-all**                             | **0.2958** | **0.2745**      | **0.3007**       | **0.2765**   | **0.3525**  | **0.4148**        | **0.4472**   |
|                                         |            |                 |                  |              |             |                   |              |
| ACC-abstract_algebra                    | 0.2200     | 0.2500          | 0.3000           | 0.2400       | 0.2500      | 0.3200            | 0.2800       |
| ACC-anatomy                             | 0.2963     | 0.2667          | 0.3333           | 0.2444       | 0.3852      | 0.4667            | 0.4889       |
| ACC-astronomy                           | 0.2961     | 0.2763          | 0.2500           | 0.2434       | 0.3487      | 0.4737            | 0.4671       |
| ACC-business_ethics                     | 0.2900     | 0.2900          | 0.3200           | 0.1900       | 0.4100      | 0.4100            | 0.4300       |
| ACC-clinical_knowledge                  | 0.2943     | 0.3208          | 0.3887           | 0.3019       | 0.3585      | 0.4113            | 0.4189       |
| ACC-college_biology                     | 0.3056     | 0.3125          | 0.3264           | 0.2153       | 0.3819      | 0.4167            | 0.4722       |
| ACC-college_chemistry                   | 0.2800     | 0.2700          | 0.2400           | 0.2300       | 0.2900      | 0.2800            | 0.2400       |
| ACC-college_computer_science            | 0.3100     | 0.3100          | 0.3100           | 0.3000       | 0.2900      | 0.4000            | 0.3700       |
| ACC-college_mathematics                 | 0.2900     | 0.2500          | 0.2800           | 0.2900       | 0.3400      | 0.3200            | 0.2500       |
| ACC-college_medicine                    | 0.2890     | 0.2659          | 0.3179           | 0.2659       | 0.3237      | 0.3699            | 0.4220       |
| ACC-college_physics                     | 0.2157     | 0.2451          | 0.1863           | 0.2157       | 0.2451      | 0.2549            | 0.1863       |
| ACC-computer_security                   | 0.3100     | 0.3600          | 0.3800           | 0.2800       | 0.4500      | 0.5400            | 0.6300       |
| ACC-conceptual_physics                  | 0.3362     | 0.2723          | 0.3064           | 0.3149       | 0.3702      | 0.3574            | 0.3915       |
| ACC-econometrics                        | 0.2895     | 0.2368          | 0.2895           | 0.2632       | 0.2632      | 0.3070            | 0.2719       |
| ACC-electrical_engineering              | 0.2897     | 0.3034          | 0.3034           | 0.2828       | 0.2483      | 0.4966            | 0.3862       |
| ACC-elementary_mathematics              | 0.2698     | 0.2646          | 0.2698           | 0.2593       | 0.2646      | 0.2487            | 0.2487       |
| ACC-formal_logic                        | 0.2540     | 0.4048          | 0.2381           | 0.1905       | 0.2619      | 0.3016            | 0.3889       |
| ACC-global_facts                        | 0.2700     | 0.3200          | 0.3200           | 0.3100       | 0.3000      | 0.2900            | 0.3400       |
| ACC-high_school_biology                 | 0.3097     | 0.2484          | 0.2968           | 0.2645       | 0.3387      | 0.4290            | 0.5065       |
| ACC-high_school_chemistry               | 0.2020     | 0.2660          | 0.2512           | 0.2512       | 0.2956      | 0.3350            | 0.2660       |
| ACC-high_school_computer_science        | 0.3400     | 0.2700          | 0.2800           | 0.3200       | 0.3300      | 0.2700            | 0.4500       |
| ACC-high_school_european_history        | 0.3455     | 0.2848          | 0.3455           | 0.2909       | 0.4667      | 0.4727            | 0.6121       |
| ACC-high_school_geography               | 0.3737     | 0.3283          | 0.3333           | 0.1667       | 0.3333      | 0.4899            | 0.5000       |
| ACC-high_school_government_and_politics | 0.3782     | 0.2124          | 0.3575           | 0.2591       | 0.4611      | 0.5959            | 0.6425       |
| ACC-high_school_macroeconomics          | 0.3821     | 0.2718          | 0.3564           | 0.2615       | 0.3410      | 0.4282            | 0.4256       |
| ACC-high_school_mathematics             | 0.2778     | 0.2667          | 0.2407           | 0.2481       | 0.2630      | 0.2667            | 0.2593       |
| ACC-high_school_microeconomics          | 0.2941     | 0.3067          | 0.2941           | 0.2899       | 0.3319      | 0.4370            | 0.4454       |
| ACC-high_school_physics                 | 0.2583     | 0.2649          | 0.2517           | 0.3179       | 0.2649      | 0.2980            | 0.2517       |
| ACC-high_school_psychology              | 0.2844     | 0.3229          | 0.3505           | 0.2440       | 0.4789      | 0.5486            | 0.5835       |
| ACC-high_school_statistics              | 0.4028     | 0.2454          | 0.3981           | 0.1852       | 0.3241      | 0.2546            | 0.2685       |
| ACC-high_school_us_history              | 0.2892     | 0.2255          | 0.3137           | 0.2892       | 0.3284      | 0.5490            | 0.5343       |
| ACC-high_school_world_history           | 0.2489     | 0.2785          | 0.2869           | 0.2996       | 0.4262      | 0.5105            | 0.6287       |
| ACC-human_aging                         | 0.3274     | 0.1659          | 0.2870           | 0.4215       | 0.3991      | 0.5157            | 0.5112       |
| ACC-human_sexuality                     | 0.3511     | 0.2519          | 0.2748           | 0.2901       | 0.3435      | 0.4962            | 0.5649       |
| ACC-international_law                   | 0.3802     | 0.2231          | 0.3636           | 0.2479       | 0.5207      | 0.5207            | 0.6860       |
| ACC-jurisprudence                       | 0.3704     | 0.2315          | 0.3426           | 0.3426       | 0.4167      | 0.4444            | 0.4722       |
| ACC-logical_fallacies                   | 0.2945     | 0.2638          | 0.2883           | 0.2638       | 0.4172      | 0.4847            | 0.5031       |
| ACC-machine_learning                    | 0.3125     | 0.2232          | 0.2321           | 0.3750       | 0.2768      | 0.3571            | 0.3304       |
| ACC-management                          | 0.3301     | 0.2816          | 0.2524           | 0.2816       | 0.3301      | 0.5243            | 0.6311       |
| ACC-marketing                           | 0.3120     | 0.2735          | 0.3761           | 0.2949       | 0.4615      | 0.5897            | 0.7094       |
| ACC-medical_genetics                    | 0.3100     | 0.2400          | 0.2700           | 0.2800       | 0.3700      | 0.5100            | 0.5100       |
| ACC-miscellaneous                       | 0.3001     | 0.2899          | 0.3678           | 0.2976       | 0.4278      | 0.5900            | 0.6296       |
| ACC-moral_disputes                      | 0.2977     | 0.2659          | 0.3295           | 0.3092       | 0.4133      | 0.4798            | 0.4566       |
| ACC-moral_scenarios                     | 0.2436     | 0.2469          | 0.2469           | 0.2492       | 0.2425      | 0.2715            | 0.2480       |
| ACC-nutrition                           | 0.2810     | 0.2908          | 0.3301           | 0.2582       | 0.3922      | 0.3758            | 0.5163       |
| ACC-philosophy                          | 0.3183     | 0.2830          | 0.2830           | 0.2830       | 0.4051      | 0.4662            | 0.5145       |
| ACC-prehistory                          | 0.3056     | 0.3210          | 0.3210           | 0.3117       | 0.3519      | 0.5216            | 0.5093       |
| ACC-professional_accounting             | 0.2447     | 0.2872          | 0.2553           | 0.2979       | 0.2730      | 0.3050            | 0.3227       |
| ACC-professional_law                    | 0.2784     | 0.2705          | 0.2523           | 0.2497       | 0.2973      | 0.3064            | 0.3566       |
| ACC-professional_medicine               | 0.2206     | 0.2059          | 0.2500           | 0.3125       | 0.4265      | 0.3860            | 0.5000       |
| ACC-professional_psychology             | 0.2876     | 0.2925          | 0.2696           | 0.2647       | 0.3546      | 0.3693            | 0.4575       |
| ACC-public_relations                    | 0.3455     | 0.3182          | 0.4091           | 0.3364       | 0.4091      | 0.5273            | 0.5545       |
| ACC-security_studies                    | 0.3796     | 0.2816          | 0.2939           | 0.3102       | 0.3306      | 0.4245            | 0.5224       |
| ACC-sociology                           | 0.2239     | 0.2587          | 0.2488           | 0.3532       | 0.4726      | 0.5473            | 0.6418       |
| ACC-us_foreign_policy                   | 0.3500     | 0.3200          | 0.3900           | 0.4200       | 0.4300      | 0.6100            | 0.7200       |
| ACC-virology                            | 0.3494     | 0.2530          | 0.3494           | 0.3554       | 0.3253      | 0.4398            | 0.4096       |
| ACC-world_religions                     | 0.3158     | 0.3041          | 0.4035           | 0.3333       | 0.4912      | 0.6550            | 0.6491       |
