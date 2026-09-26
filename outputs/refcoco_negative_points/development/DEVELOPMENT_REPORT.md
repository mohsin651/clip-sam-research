# RefCOCO validation: negative-point development

Recommendation: **STOP**. Positive lower paired 95% bounds for final correct-instance, final IoU, and candidate-oracle IoU versus matched POS3; predeclared before inference. Exploratory, not multiplicity-adjusted.

500 deterministic validation images, 3689 expressions, 1296 targets; seed 2026. Eligible images: 1483. All expressions in selected images retained. Prior COCO overlap: []; RefCOCO test overlap: []. Metadata-only cohort selected before inference. Category-name text is explicitly allowed auxiliary metadata. No new testA/testB predictions.

## Main strategy results

All rates are fractions. Primary full-image expression-macro instance IoU. All unavailable-negative cases remain and exactly reuse POS3. Frozen final uses the unchanged smart selector and density fallback.

| method | iou | correct_instance | wrong_instance | p_at_05 | harm |
| --- | --- | --- | --- | --- | --- |
| CLIP | 0.1948 | 0.5069 | 0.4915 | 0.0347 | 0.0000 |
| POS3/sam_score | 0.2821 | 0.5115 | 0.4736 | 0.2245 | 0.4546 |
| POS3/smart | 0.2955 | 0.5153 | 0.4717 | 0.2372 | 0.4017 |
| POS3/final | 0.2950 | 0.5140 | 0.4757 | 0.2296 | 0.3288 |
| NEG_LOW1/sam_score | 0.2754 | 0.5053 | 0.4841 | 0.2155 | 0.4549 |
| NEG_LOW1/smart | 0.2932 | 0.5091 | 0.4798 | 0.2361 | 0.4023 |
| NEG_LOW1/final | 0.2936 | 0.5123 | 0.4787 | 0.2299 | 0.3361 |
| NEG_LOW2/sam_score | 0.2736 | 0.5121 | 0.4776 | 0.2066 | 0.4497 |
| NEG_LOW2/smart | 0.2863 | 0.5102 | 0.4782 | 0.2193 | 0.4085 |
| NEG_LOW2/final | 0.2858 | 0.5110 | 0.4793 | 0.2128 | 0.3418 |
| NEG_CONTRAST1/sam_score | 0.2731 | 0.5096 | 0.4793 | 0.2117 | 0.4614 |
| NEG_CONTRAST1/smart | 0.2885 | 0.5096 | 0.4793 | 0.2280 | 0.4112 |
| NEG_CONTRAST1/final | 0.2886 | 0.5083 | 0.4825 | 0.2215 | 0.3432 |
| NEG_CONTRAST2/sam_score | 0.2691 | 0.5050 | 0.4844 | 0.2033 | 0.4619 |
| NEG_CONTRAST2/smart | 0.2859 | 0.5099 | 0.4790 | 0.2217 | 0.4072 |
| NEG_CONTRAST2/final | 0.2855 | 0.5075 | 0.4831 | 0.2147 | 0.3351 |
| NEG_REGION1/sam_score | 0.2776 | 0.5118 | 0.4747 | 0.2212 | 0.4649 |
| NEG_REGION1/smart | 0.2929 | 0.5140 | 0.4733 | 0.2342 | 0.4058 |
| NEG_REGION1/final | 0.2925 | 0.5112 | 0.4782 | 0.2274 | 0.3356 |
| NEG_REGION2/sam_score | 0.2760 | 0.5102 | 0.4755 | 0.2169 | 0.4676 |
| NEG_REGION2/smart | 0.2912 | 0.5134 | 0.4741 | 0.2293 | 0.4118 |
| NEG_REGION2/final | 0.2909 | 0.5102 | 0.4801 | 0.2223 | 0.3397 |

## Paired contrasts versus matched POS3

2,000 image-level resamples, seed 2026; all expressions stay grouped. No multiplicity adjustment.

| method | reference | metric | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| NEG_LOW1/sam_score | POS3/sam_score | iou | -0.0067 | -0.0116 | -0.0022 |
| NEG_LOW1/sam_score | POS3/sam_score | correct_instance | -0.0062 | -0.0157 | 0.0030 |
| NEG_LOW1/sam_score | POS3/sam_score | p_at_05 | -0.0089 | -0.0185 | 0.0003 |
| NEG_LOW1/sam_score | POS3/sam_score | harm | 0.0003 | -0.0123 | 0.0127 |
| NEG_LOW1/smart | POS3/smart | iou | -0.0022 | -0.0057 | 0.0013 |
| NEG_LOW1/smart | POS3/smart | correct_instance | -0.0062 | -0.0146 | 0.0019 |
| NEG_LOW1/smart | POS3/smart | p_at_05 | -0.0011 | -0.0096 | 0.0076 |
| NEG_LOW1/smart | POS3/smart | harm | 0.0005 | -0.0093 | 0.0103 |
| NEG_LOW1/final | POS3/final | iou | -0.0014 | -0.0046 | 0.0020 |
| NEG_LOW1/final | POS3/final | correct_instance | -0.0016 | -0.0101 | 0.0069 |
| NEG_LOW1/final | POS3/final | p_at_05 | 0.0003 | -0.0081 | 0.0086 |
| NEG_LOW1/final | POS3/final | harm | 0.0073 | -0.0027 | 0.0177 |
| NEG_LOW2/sam_score | POS3/sam_score | iou | -0.0084 | -0.0139 | -0.0033 |
| NEG_LOW2/sam_score | POS3/sam_score | correct_instance | 0.0005 | -0.0103 | 0.0108 |
| NEG_LOW2/sam_score | POS3/sam_score | p_at_05 | -0.0179 | -0.0293 | -0.0073 |
| NEG_LOW2/sam_score | POS3/sam_score | harm | -0.0049 | -0.0186 | 0.0091 |
| NEG_LOW2/smart | POS3/smart | iou | -0.0091 | -0.0132 | -0.0048 |
| NEG_LOW2/smart | POS3/smart | correct_instance | -0.0052 | -0.0149 | 0.0043 |
| NEG_LOW2/smart | POS3/smart | p_at_05 | -0.0179 | -0.0275 | -0.0081 |
| NEG_LOW2/smart | POS3/smart | harm | 0.0068 | -0.0046 | 0.0186 |
| NEG_LOW2/final | POS3/final | iou | -0.0092 | -0.0131 | -0.0050 |
| NEG_LOW2/final | POS3/final | correct_instance | -0.0030 | -0.0128 | 0.0064 |
| NEG_LOW2/final | POS3/final | p_at_05 | -0.0168 | -0.0263 | -0.0074 |
| NEG_LOW2/final | POS3/final | harm | 0.0130 | 0.0013 | 0.0251 |
| NEG_CONTRAST1/sam_score | POS3/sam_score | iou | -0.0089 | -0.0127 | -0.0051 |
| NEG_CONTRAST1/sam_score | POS3/sam_score | correct_instance | -0.0019 | -0.0105 | 0.0072 |
| NEG_CONTRAST1/sam_score | POS3/sam_score | p_at_05 | -0.0127 | -0.0207 | -0.0052 |
| NEG_CONTRAST1/sam_score | POS3/sam_score | harm | 0.0068 | -0.0027 | 0.0163 |
| NEG_CONTRAST1/smart | POS3/smart | iou | -0.0069 | -0.0103 | -0.0038 |
| NEG_CONTRAST1/smart | POS3/smart | correct_instance | -0.0057 | -0.0134 | 0.0017 |
| NEG_CONTRAST1/smart | POS3/smart | p_at_05 | -0.0092 | -0.0174 | -0.0017 |
| NEG_CONTRAST1/smart | POS3/smart | harm | 0.0095 | 0.0011 | 0.0183 |
| NEG_CONTRAST1/final | POS3/final | iou | -0.0064 | -0.0096 | -0.0033 |
| NEG_CONTRAST1/final | POS3/final | correct_instance | -0.0057 | -0.0136 | 0.0016 |
| NEG_CONTRAST1/final | POS3/final | p_at_05 | -0.0081 | -0.0160 | -0.0005 |
| NEG_CONTRAST1/final | POS3/final | harm | 0.0144 | 0.0053 | 0.0239 |
| NEG_CONTRAST2/sam_score | POS3/sam_score | iou | -0.0130 | -0.0175 | -0.0087 |
| NEG_CONTRAST2/sam_score | POS3/sam_score | correct_instance | -0.0065 | -0.0159 | 0.0027 |
| NEG_CONTRAST2/sam_score | POS3/sam_score | p_at_05 | -0.0211 | -0.0300 | -0.0128 |
| NEG_CONTRAST2/sam_score | POS3/sam_score | harm | 0.0073 | -0.0038 | 0.0187 |
| NEG_CONTRAST2/smart | POS3/smart | iou | -0.0095 | -0.0137 | -0.0055 |
| NEG_CONTRAST2/smart | POS3/smart | correct_instance | -0.0054 | -0.0143 | 0.0027 |
| NEG_CONTRAST2/smart | POS3/smart | p_at_05 | -0.0155 | -0.0242 | -0.0069 |
| NEG_CONTRAST2/smart | POS3/smart | harm | 0.0054 | -0.0041 | 0.0149 |
| NEG_CONTRAST2/final | POS3/final | iou | -0.0095 | -0.0135 | -0.0057 |
| NEG_CONTRAST2/final | POS3/final | correct_instance | -0.0065 | -0.0154 | 0.0019 |
| NEG_CONTRAST2/final | POS3/final | p_at_05 | -0.0149 | -0.0237 | -0.0065 |
| NEG_CONTRAST2/final | POS3/final | harm | 0.0062 | -0.0045 | 0.0174 |
| NEG_REGION1/sam_score | POS3/sam_score | iou | -0.0044 | -0.0071 | -0.0017 |
| NEG_REGION1/sam_score | POS3/sam_score | correct_instance | 0.0003 | -0.0045 | 0.0048 |
| NEG_REGION1/sam_score | POS3/sam_score | p_at_05 | -0.0033 | -0.0076 | 0.0016 |
| NEG_REGION1/sam_score | POS3/sam_score | harm | 0.0103 | 0.0045 | 0.0163 |
| NEG_REGION1/smart | POS3/smart | iou | -0.0026 | -0.0050 | -0.0002 |
| NEG_REGION1/smart | POS3/smart | correct_instance | -0.0014 | -0.0048 | 0.0021 |
| NEG_REGION1/smart | POS3/smart | p_at_05 | -0.0030 | -0.0072 | 0.0011 |
| NEG_REGION1/smart | POS3/smart | harm | 0.0041 | -0.0011 | 0.0091 |
| NEG_REGION1/final | POS3/final | iou | -0.0025 | -0.0048 | -0.0002 |
| NEG_REGION1/final | POS3/final | correct_instance | -0.0027 | -0.0066 | 0.0011 |
| NEG_REGION1/final | POS3/final | p_at_05 | -0.0022 | -0.0062 | 0.0019 |
| NEG_REGION1/final | POS3/final | harm | 0.0068 | 0.0011 | 0.0125 |
| NEG_REGION2/sam_score | POS3/sam_score | iou | -0.0060 | -0.0087 | -0.0034 |
| NEG_REGION2/sam_score | POS3/sam_score | correct_instance | -0.0014 | -0.0059 | 0.0032 |
| NEG_REGION2/sam_score | POS3/sam_score | p_at_05 | -0.0076 | -0.0121 | -0.0032 |
| NEG_REGION2/sam_score | POS3/sam_score | harm | 0.0130 | 0.0070 | 0.0193 |
| NEG_REGION2/smart | POS3/smart | iou | -0.0042 | -0.0068 | -0.0017 |
| NEG_REGION2/smart | POS3/smart | correct_instance | -0.0019 | -0.0065 | 0.0026 |
| NEG_REGION2/smart | POS3/smart | p_at_05 | -0.0079 | -0.0128 | -0.0032 |
| NEG_REGION2/smart | POS3/smart | harm | 0.0100 | 0.0045 | 0.0154 |
| NEG_REGION2/final | POS3/final | iou | -0.0041 | -0.0064 | -0.0017 |
| NEG_REGION2/final | POS3/final | correct_instance | -0.0038 | -0.0086 | 0.0010 |
| NEG_REGION2/final | POS3/final | p_at_05 | -0.0073 | -0.0118 | -0.0030 |
| NEG_REGION2/final | POS3/final | harm | 0.0108 | 0.0050 | 0.0167 |

## Same-category ambiguity

| method | expressions | iou | correct_instance | wrong_instance | p_at_05 |
| --- | --- | --- | --- | --- | --- |
| CLIP | 3689 | 0.1948 | 0.5069 | 0.4915 | 0.0347 |
| POS3/sam_score | 3689 | 0.2821 | 0.5115 | 0.4736 | 0.2245 |
| POS3/smart | 3689 | 0.2955 | 0.5153 | 0.4717 | 0.2372 |
| POS3/final | 3689 | 0.2950 | 0.5140 | 0.4757 | 0.2296 |
| NEG_LOW1/sam_score | 3689 | 0.2754 | 0.5053 | 0.4841 | 0.2155 |
| NEG_LOW1/smart | 3689 | 0.2932 | 0.5091 | 0.4798 | 0.2361 |
| NEG_LOW1/final | 3689 | 0.2936 | 0.5123 | 0.4787 | 0.2299 |
| NEG_LOW2/sam_score | 3689 | 0.2736 | 0.5121 | 0.4776 | 0.2066 |
| NEG_LOW2/smart | 3689 | 0.2863 | 0.5102 | 0.4782 | 0.2193 |
| NEG_LOW2/final | 3689 | 0.2858 | 0.5110 | 0.4793 | 0.2128 |
| NEG_CONTRAST1/sam_score | 3689 | 0.2731 | 0.5096 | 0.4793 | 0.2117 |
| NEG_CONTRAST1/smart | 3689 | 0.2885 | 0.5096 | 0.4793 | 0.2280 |
| NEG_CONTRAST1/final | 3689 | 0.2886 | 0.5083 | 0.4825 | 0.2215 |
| NEG_CONTRAST2/sam_score | 3689 | 0.2691 | 0.5050 | 0.4844 | 0.2033 |
| NEG_CONTRAST2/smart | 3689 | 0.2859 | 0.5099 | 0.4790 | 0.2217 |
| NEG_CONTRAST2/final | 3689 | 0.2855 | 0.5075 | 0.4831 | 0.2147 |
| NEG_REGION1/sam_score | 3689 | 0.2776 | 0.5118 | 0.4747 | 0.2212 |
| NEG_REGION1/smart | 3689 | 0.2929 | 0.5140 | 0.4733 | 0.2342 |
| NEG_REGION1/final | 3689 | 0.2925 | 0.5112 | 0.4782 | 0.2274 |
| NEG_REGION2/sam_score | 3689 | 0.2760 | 0.5102 | 0.4755 | 0.2169 |
| NEG_REGION2/smart | 3689 | 0.2912 | 0.5134 | 0.4741 | 0.2293 |
| NEG_REGION2/final | 3689 | 0.2909 | 0.5102 | 0.4801 | 0.2223 |

## Negative-point availability and outcomes

Improvement/harm here is directly against matched POS3, while the main table harm is against CLIP. Unavailable cases are in every expression denominator.

| strategy | selector | available | no_negative_available | sam_improved | sam_harmed | correct_instance_improved | correct_instance_degraded |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NEG_LOW1 | sam_score | 0.9851 | 0.0149 | 0.3912 | 0.4633 | 0.0314 | 0.0377 |
| NEG_LOW1 | smart | 0.9851 | 0.0149 | 0.4050 | 0.4576 | 0.0271 | 0.0333 |
| NEG_LOW1 | final | 0.9851 | 0.0149 | 0.3706 | 0.4283 | 0.0298 | 0.0314 |
| NEG_LOW2 | sam_score | 0.9555 | 0.0445 | 0.3790 | 0.4587 | 0.0426 | 0.0420 |
| NEG_LOW2 | smart | 0.9555 | 0.0445 | 0.3763 | 0.4673 | 0.0380 | 0.0431 |
| NEG_LOW2 | final | 0.9555 | 0.0445 | 0.3489 | 0.4372 | 0.0390 | 0.0420 |
| NEG_CONTRAST1 | sam_score | 0.8124 | 0.1876 | 0.3196 | 0.3836 | 0.0320 | 0.0339 |
| NEG_CONTRAST1 | smart | 0.8124 | 0.1876 | 0.3069 | 0.4015 | 0.0249 | 0.0306 |
| NEG_CONTRAST1 | final | 0.8124 | 0.1876 | 0.2792 | 0.3716 | 0.0252 | 0.0309 |
| NEG_CONTRAST2 | sam_score | 0.7563 | 0.2437 | 0.2917 | 0.3635 | 0.0350 | 0.0415 |
| NEG_CONTRAST2 | smart | 0.7563 | 0.2437 | 0.2860 | 0.3722 | 0.0293 | 0.0347 |
| NEG_CONTRAST2 | final | 0.7563 | 0.2437 | 0.2627 | 0.3448 | 0.0306 | 0.0371 |
| NEG_REGION1 | sam_score | 0.1624 | 0.8376 | 0.0629 | 0.0786 | 0.0098 | 0.0095 |
| NEG_REGION1 | smart | 0.1624 | 0.8376 | 0.0651 | 0.0748 | 0.0060 | 0.0073 |
| NEG_REGION1 | final | 0.1624 | 0.8376 | 0.0621 | 0.0754 | 0.0068 | 0.0095 |
| NEG_REGION2 | sam_score | 0.1326 | 0.8674 | 0.0474 | 0.0699 | 0.0092 | 0.0106 |
| NEG_REGION2 | smart | 0.1326 | 0.8674 | 0.0510 | 0.0634 | 0.0081 | 0.0100 |
| NEG_REGION2 | final | 0.1326 | 0.8674 | 0.0483 | 0.0632 | 0.0076 | 0.0114 |

## Post-hoc point accuracy

Point-level fractions condition on emitted points. Expression target-hit rates are separately retained in diagnostic_summary.csv. Exclusive priority: target, same-category competitor, different-category, same-category crowd, background. Same-category crowds are separately flagged rather than mislabeled as unannotated background; they are not individual-instance competitors. Raw labels also retain overlapping membership flags; overlap can make object identity ambiguous. No labels entered prompting.

| strategy | negative_points | point_fraction_same_category_competitor | point_fraction_target | point_fraction_different_category | point_fraction_same_category_crowd | point_fraction_background | target_hit_given_available |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NEG_LOW1 | 3634 | 0.1676 | 0.0454 | 0.1478 | 0.0019 | 0.6373 | 0.0454 |
| NEG_LOW2 | 7050 | 0.1682 | 0.0458 | 0.1593 | 0.0033 | 0.6234 | 0.0817 |
| NEG_CONTRAST1 | 2997 | 0.4224 | 0.1415 | 0.1428 | 0.0033 | 0.2900 | 0.1415 |
| NEG_CONTRAST2 | 5580 | 0.3771 | 0.1427 | 0.1341 | 0.0063 | 0.3400 | 0.2530 |
| NEG_REGION1 | 599 | 0.3639 | 0.2938 | 0.1135 | 0.0067 | 0.2220 | 0.2938 |
| NEG_REGION2 | 978 | 0.4008 | 0.2975 | 0.0982 | 0.0061 | 0.1973 | 0.4499 |

## Candidate and prompt oracles

Analysis-only best masks/strategies cannot be deployed. Candidate oracles compare to POS3 candidate oracle; fixed-selector strategy oracles compare to POS3 with that selector. All cases, including unavailable negatives, remain.

| oracle | iou | ci_low | ci_high | delta_vs_matched_pos3 | delta_ci_low | delta_ci_high |
| --- | --- | --- | --- | --- | --- | --- |
| POS3 | 0.3628 | 0.3470 | 0.3796 | 0.0000 | 0.0000 | 0.0000 |
| NEG_LOW1 | 0.3519 | 0.3367 | 0.3683 | -0.0109 | -0.0142 | -0.0077 |
| NEG_LOW2 | 0.3372 | 0.3230 | 0.3528 | -0.0256 | -0.0306 | -0.0209 |
| NEG_CONTRAST1 | 0.3523 | 0.3370 | 0.3687 | -0.0105 | -0.0135 | -0.0077 |
| NEG_CONTRAST2 | 0.3405 | 0.3264 | 0.3566 | -0.0223 | -0.0267 | -0.0180 |
| NEG_REGION1 | 0.3580 | 0.3419 | 0.3745 | -0.0049 | -0.0070 | -0.0029 |
| NEG_REGION2 | 0.3541 | 0.3387 | 0.3705 | -0.0087 | -0.0114 | -0.0060 |
| combined_prompt_candidate_oracle | 0.3962 | 0.3805 | 0.4129 | 0.0333 | 0.0307 | 0.0362 |
| combined_with_clip_oracle | 0.4227 | 0.4075 | 0.4388 | 0.0599 | 0.0561 | 0.0637 |
| negative_strategy_oracle_sam_score | 0.3330 | 0.3185 | 0.3487 | 0.0510 | 0.0463 | 0.0556 |
| negative_strategy_oracle_smart | 0.3380 | 0.3239 | 0.3537 | 0.0425 | 0.0387 | 0.0464 |
| negative_strategy_oracle_final | 0.3373 | 0.3234 | 0.3529 | 0.0423 | 0.0385 | 0.0461 |
| strategy_oracle_including_pos3_sam_score | 0.3346 | 0.3200 | 0.3502 | 0.0526 | 0.0480 | 0.0571 |
| strategy_oracle_including_pos3_smart | 0.3391 | 0.3249 | 0.3549 | 0.0436 | 0.0400 | 0.0475 |
| strategy_oracle_including_pos3_final | 0.3383 | 0.3243 | 0.3538 | 0.0433 | 0.0395 | 0.0470 |

## Expression subgroups

All methods, paired intervals, lengths, spatial/attribute/color/clothing groups are in subgroup_results.csv. Below retains final systems for readability. Lexicons unchanged from RefCOCO frozen evaluation; overlapping and confounded, not causal.

| family | group | method | expressions | iou | correct_instance | delta_iou | delta_correct_instance | harm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| length_group | long | POS3/final | 347 | 0.1982 | 0.4092 | 0.0000 | 0.0000 | 0.3804 |
| length_group | long | NEG_LOW1/final | 347 | 0.1985 | 0.4035 | 0.0003 | -0.0058 | 0.3602 |
| length_group | long | NEG_LOW2/final | 347 | 0.2015 | 0.4150 | 0.0034 | 0.0058 | 0.3545 |
| length_group | long | NEG_CONTRAST1/final | 347 | 0.1956 | 0.4035 | -0.0026 | -0.0058 | 0.3862 |
| length_group | long | NEG_CONTRAST2/final | 347 | 0.1965 | 0.4063 | -0.0017 | -0.0029 | 0.3545 |
| length_group | long | NEG_REGION1/final | 347 | 0.1972 | 0.4006 | -0.0010 | -0.0086 | 0.3890 |
| length_group | long | NEG_REGION2/final | 347 | 0.1939 | 0.3948 | -0.0043 | -0.0144 | 0.3948 |
| length_group | medium | POS3/final | 1178 | 0.2440 | 0.4643 | 0.0000 | 0.0000 | 0.3676 |
| length_group | medium | NEG_LOW1/final | 1178 | 0.2448 | 0.4660 | 0.0009 | 0.0017 | 0.3752 |
| length_group | medium | NEG_LOW2/final | 1178 | 0.2390 | 0.4694 | -0.0050 | 0.0051 | 0.3727 |
| length_group | medium | NEG_CONTRAST1/final | 1178 | 0.2432 | 0.4694 | -0.0008 | 0.0051 | 0.3735 |
| length_group | medium | NEG_CONTRAST2/final | 1178 | 0.2402 | 0.4660 | -0.0038 | 0.0017 | 0.3565 |
| length_group | medium | NEG_REGION1/final | 1178 | 0.2426 | 0.4601 | -0.0013 | -0.0042 | 0.3727 |
| length_group | medium | NEG_REGION2/final | 1178 | 0.2404 | 0.4584 | -0.0036 | -0.0059 | 0.3786 |
| length_group | short | POS3/final | 2164 | 0.3383 | 0.5578 | 0.0000 | 0.0000 | 0.2994 |
| length_group | short | NEG_LOW1/final | 2164 | 0.3354 | 0.5550 | -0.0030 | -0.0028 | 0.3110 |
| length_group | short | NEG_LOW2/final | 2164 | 0.3248 | 0.5490 | -0.0136 | -0.0088 | 0.3230 |
| length_group | short | NEG_CONTRAST1/final | 2164 | 0.3283 | 0.5462 | -0.0100 | -0.0116 | 0.3198 |
| length_group | short | NEG_CONTRAST2/final | 2164 | 0.3245 | 0.5462 | -0.0138 | -0.0116 | 0.3202 |
| length_group | short | NEG_REGION1/final | 2164 | 0.3350 | 0.5568 | -0.0034 | -0.0009 | 0.3068 |
| length_group | short | NEG_REGION2/final | 2164 | 0.3340 | 0.5568 | -0.0043 | -0.0009 | 0.3096 |
| spatial | False | POS3/final | 1697 | 0.3723 | 0.6364 | 0.0000 | 0.0000 | 0.2852 |
| spatial | False | NEG_LOW1/final | 1697 | 0.3699 | 0.6341 | -0.0024 | -0.0024 | 0.2946 |
| spatial | False | NEG_LOW2/final | 1697 | 0.3589 | 0.6364 | -0.0134 | 0.0000 | 0.3070 |
| spatial | False | NEG_CONTRAST1/final | 1697 | 0.3621 | 0.6311 | -0.0102 | -0.0053 | 0.3023 |
| spatial | False | NEG_CONTRAST2/final | 1697 | 0.3566 | 0.6352 | -0.0157 | -0.0012 | 0.3017 |
| spatial | False | NEG_REGION1/final | 1697 | 0.3705 | 0.6382 | -0.0018 | 0.0018 | 0.2852 |
| spatial | False | NEG_REGION2/final | 1697 | 0.3674 | 0.6352 | -0.0049 | -0.0012 | 0.2958 |
| spatial | True | POS3/final | 1992 | 0.2292 | 0.4096 | 0.0000 | 0.0000 | 0.3660 |
| spatial | True | NEG_LOW1/final | 1992 | 0.2285 | 0.4086 | -0.0006 | -0.0010 | 0.3715 |
| spatial | True | NEG_LOW2/final | 1992 | 0.2235 | 0.4041 | -0.0057 | -0.0055 | 0.3715 |
| spatial | True | NEG_CONTRAST1/final | 1992 | 0.2260 | 0.4036 | -0.0032 | -0.0060 | 0.3780 |
| spatial | True | NEG_CONTRAST2/final | 1992 | 0.2250 | 0.3986 | -0.0042 | -0.0110 | 0.3635 |
| spatial | True | NEG_REGION1/final | 1992 | 0.2261 | 0.4031 | -0.0030 | -0.0065 | 0.3785 |
| spatial | True | NEG_REGION2/final | 1992 | 0.2258 | 0.4036 | -0.0034 | -0.0060 | 0.3770 |
| attribute | False | POS3/final | 2671 | 0.2766 | 0.4744 | 0.0000 | 0.0000 | 0.3493 |
| attribute | False | NEG_LOW1/final | 2671 | 0.2746 | 0.4744 | -0.0020 | 0.0000 | 0.3594 |
| attribute | False | NEG_LOW2/final | 2671 | 0.2665 | 0.4699 | -0.0101 | -0.0045 | 0.3669 |
| attribute | False | NEG_CONTRAST1/final | 2671 | 0.2702 | 0.4710 | -0.0064 | -0.0034 | 0.3673 |
| attribute | False | NEG_CONTRAST2/final | 2671 | 0.2680 | 0.4691 | -0.0086 | -0.0052 | 0.3579 |
| attribute | False | NEG_REGION1/final | 2671 | 0.2731 | 0.4702 | -0.0035 | -0.0041 | 0.3560 |
| attribute | False | NEG_REGION2/final | 2671 | 0.2714 | 0.4706 | -0.0052 | -0.0037 | 0.3620 |
| attribute | True | POS3/final | 1018 | 0.3433 | 0.6179 | 0.0000 | 0.0000 | 0.2750 |
| attribute | True | NEG_LOW1/final | 1018 | 0.3434 | 0.6120 | 0.0001 | -0.0059 | 0.2750 |
| attribute | True | NEG_LOW2/final | 1018 | 0.3364 | 0.6189 | -0.0069 | 0.0010 | 0.2760 |
| attribute | True | NEG_CONTRAST1/final | 1018 | 0.3369 | 0.6061 | -0.0065 | -0.0118 | 0.2800 |
| attribute | True | NEG_CONTRAST2/final | 1018 | 0.3314 | 0.6081 | -0.0119 | -0.0098 | 0.2750 |
| attribute | True | NEG_REGION1/final | 1018 | 0.3436 | 0.6189 | 0.0003 | 0.0010 | 0.2819 |
| attribute | True | NEG_REGION2/final | 1018 | 0.3422 | 0.6139 | -0.0011 | -0.0039 | 0.2809 |
| color | False | POS3/final | 2809 | 0.2797 | 0.4781 | 0.0000 | 0.0000 | 0.3496 |
| color | False | NEG_LOW1/final | 2809 | 0.2781 | 0.4813 | -0.0016 | 0.0032 | 0.3574 |
| color | False | NEG_LOW2/final | 2809 | 0.2699 | 0.4767 | -0.0098 | -0.0014 | 0.3660 |
| color | False | NEG_CONTRAST1/final | 2809 | 0.2736 | 0.4760 | -0.0061 | -0.0021 | 0.3649 |
| color | False | NEG_CONTRAST2/final | 2809 | 0.2707 | 0.4728 | -0.0090 | -0.0053 | 0.3567 |
| color | False | NEG_REGION1/final | 2809 | 0.2764 | 0.4742 | -0.0033 | -0.0039 | 0.3556 |
| color | False | NEG_REGION2/final | 2809 | 0.2747 | 0.4742 | -0.0050 | -0.0039 | 0.3613 |
| color | True | POS3/final | 880 | 0.3439 | 0.6284 | 0.0000 | 0.0000 | 0.2625 |
| color | True | NEG_LOW1/final | 880 | 0.3431 | 0.6114 | -0.0009 | -0.0170 | 0.2682 |
| color | True | NEG_LOW2/final | 880 | 0.3365 | 0.6205 | -0.0075 | -0.0080 | 0.2648 |
| color | True | NEG_CONTRAST1/final | 880 | 0.3367 | 0.6114 | -0.0073 | -0.0170 | 0.2739 |
| color | True | NEG_CONTRAST2/final | 880 | 0.3329 | 0.6182 | -0.0110 | -0.0102 | 0.2659 |
| color | True | NEG_REGION1/final | 880 | 0.3440 | 0.6295 | 0.0000 | 0.0011 | 0.2716 |
| color | True | NEG_REGION2/final | 880 | 0.3429 | 0.6250 | -0.0011 | -0.0034 | 0.2705 |
| clothing | False | POS3/final | 3342 | 0.2894 | 0.5027 | 0.0000 | 0.0000 | 0.3318 |
| clothing | False | NEG_LOW1/final | 3342 | 0.2881 | 0.5009 | -0.0013 | -0.0018 | 0.3399 |
| clothing | False | NEG_LOW2/final | 3342 | 0.2800 | 0.4988 | -0.0093 | -0.0039 | 0.3465 |
| clothing | False | NEG_CONTRAST1/final | 3342 | 0.2830 | 0.4961 | -0.0064 | -0.0066 | 0.3483 |
| clothing | False | NEG_CONTRAST2/final | 3342 | 0.2800 | 0.4958 | -0.0094 | -0.0069 | 0.3402 |
| clothing | False | NEG_REGION1/final | 3342 | 0.2871 | 0.4997 | -0.0023 | -0.0030 | 0.3378 |
| clothing | False | NEG_REGION2/final | 3342 | 0.2854 | 0.4991 | -0.0040 | -0.0036 | 0.3432 |
| clothing | True | POS3/final | 347 | 0.3493 | 0.6225 | 0.0000 | 0.0000 | 0.2997 |
| clothing | True | NEG_LOW1/final | 347 | 0.3462 | 0.6225 | -0.0031 | 0.0000 | 0.2997 |
| clothing | True | NEG_LOW2/final | 347 | 0.3410 | 0.6282 | -0.0083 | 0.0058 | 0.2968 |
| clothing | True | NEG_CONTRAST1/final | 347 | 0.3431 | 0.6254 | -0.0062 | 0.0029 | 0.2939 |
| clothing | True | NEG_CONTRAST2/final | 347 | 0.3392 | 0.6196 | -0.0101 | -0.0029 | 0.2853 |
| clothing | True | NEG_REGION1/final | 347 | 0.3446 | 0.6225 | -0.0046 | 0.0000 | 0.3141 |
| clothing | True | NEG_REGION2/final | 347 | 0.3443 | 0.6167 | -0.0049 | -0.0058 | 0.3055 |

## Harm distributions versus CLIP

| method | improved | harm | unchanged | median_delta | mean_positive_delta | mean_negative_delta | worst_decile_delta |
| --- | --- | --- | --- | --- | --- | --- | --- |
| CLIP | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| POS3/sam_score | 0.5248 | 0.4546 | 0.0206 | 0.0145 | 0.2654 | -0.1144 | -0.2611 |
| POS3/smart | 0.5771 | 0.4017 | 0.0211 | 0.0450 | 0.2448 | -0.1011 | -0.2220 |
| POS3/final | 0.5351 | 0.3288 | 0.1361 | 0.0241 | 0.2485 | -0.0997 | -0.2070 |
| NEG_LOW1/sam_score | 0.5237 | 0.4549 | 0.0214 | 0.0126 | 0.2500 | -0.1107 | -0.2535 |
| NEG_LOW1/smart | 0.5763 | 0.4023 | 0.0214 | 0.0442 | 0.2402 | -0.0995 | -0.2207 |
| NEG_LOW1/final | 0.5430 | 0.3361 | 0.1209 | 0.0261 | 0.2429 | -0.0986 | -0.2068 |
| NEG_LOW2/sam_score | 0.5291 | 0.4497 | 0.0211 | 0.0119 | 0.2418 | -0.1093 | -0.2482 |
| NEG_LOW2/smart | 0.5698 | 0.4085 | 0.0217 | 0.0405 | 0.2329 | -0.1008 | -0.2253 |
| NEG_LOW2/final | 0.5283 | 0.3418 | 0.1298 | 0.0174 | 0.2367 | -0.0998 | -0.2126 |
| NEG_CONTRAST1/sam_score | 0.5186 | 0.4614 | 0.0201 | 0.0082 | 0.2537 | -0.1154 | -0.2622 |
| NEG_CONTRAST1/smart | 0.5684 | 0.4112 | 0.0203 | 0.0410 | 0.2382 | -0.1013 | -0.2242 |
| NEG_CONTRAST1/final | 0.5310 | 0.3432 | 0.1258 | 0.0187 | 0.2415 | -0.1004 | -0.2100 |
| NEG_CONTRAST2/sam_score | 0.5169 | 0.4619 | 0.0211 | 0.0085 | 0.2452 | -0.1137 | -0.2605 |
| NEG_CONTRAST2/smart | 0.5717 | 0.4072 | 0.0211 | 0.0410 | 0.2312 | -0.1009 | -0.2241 |
| NEG_CONTRAST2/final | 0.5327 | 0.3351 | 0.1323 | 0.0186 | 0.2335 | -0.1005 | -0.2092 |
| NEG_REGION1/sam_score | 0.5150 | 0.4649 | 0.0201 | 0.0079 | 0.2645 | -0.1150 | -0.2648 |
| NEG_REGION1/smart | 0.5731 | 0.4058 | 0.0211 | 0.0414 | 0.2426 | -0.1009 | -0.2236 |
| NEG_REGION1/final | 0.5327 | 0.3356 | 0.1317 | 0.0187 | 0.2461 | -0.0995 | -0.2086 |
| NEG_REGION2/sam_score | 0.5123 | 0.4676 | 0.0201 | 0.0064 | 0.2634 | -0.1149 | -0.2631 |
| NEG_REGION2/smart | 0.5671 | 0.4118 | 0.0211 | 0.0383 | 0.2435 | -0.1013 | -0.2246 |
| NEG_REGION2/final | 0.5267 | 0.3397 | 0.1336 | 0.0163 | 0.2470 | -0.1000 | -0.2107 |

## Ten development answers

1. Largest observed mIoU gain: NEG_LOW1/final, -0.0014 [-0.0046, +0.0020]. This is a descriptive development ranking, not a test claim.
2. Largest correct-instance gain: NEG_LOW2/sam_score, +0.0005 [-0.0103, +0.0108]. See every negative variant and matched-selector interval above.
3. Predeclared joint decision: STOP; selected strategy: None. Descriptive best rows alone do not determine continuation.
4. NEG_CONTRAST same-category effects are retained in the ambiguity table and subgroup CSV, including negative results. A generic category map is not a guarantee of a true competitor.
5. True competitor hit fractions are reported per emitted point above, separately from negative availability.
6. Target hits and conditional expression target-hit rates above quantify bad negative cues; post-hoc target hits never change a prediction.
7. Every strategy candidate oracle and its paired change is reported. A fixed-selector improvement alone is insufficient for the predeclared decision.
8. Length, spatial, attribute, color and clothing subgroup changes are disclosed; no subgroup changed rules.
9. Both harm versus CLIP and direct harm versus POS3 are retained; unchanged unavailable/fallback cases remain in denominators.
10. Follow the predeclared decision above. Even a positive development result requires independent evaluation; no testA/testB run is authorized here. No formula search or selector refitting followed these results.

## Reproduction and integrity

Protocol: ../../../NEGATIVE_POINTS_PLAN.md. Scripts: prepare_negative_points.py, run_negative_points.py --smoke, run_negative_points.py --resume, evaluate_negative_points.py, report_negative_points.py, visualize_negative_points.py, audit_negative_points.py. Use the existing environment. Never overwrite a completed run; resume checks exact source/input signatures. 52 tests passed after resolving Windows sandbox temp permissions. Exactly 20 technical samples preceded full inference without GT performance evaluation. Existing frozen functions were imported unchanged; positive feature semantics remain unchanged.

[Representative panels](examples.html). Full masks, features, points, inferred support regions, raw expressions and provenance are retained. Final audit is in audit.json; wall-time fields include hashing/loading/saving and are not pure GPU throughput.
