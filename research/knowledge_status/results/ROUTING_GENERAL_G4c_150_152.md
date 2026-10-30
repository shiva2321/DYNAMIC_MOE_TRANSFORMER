# Lab G4c, a robust calibration against wrongly verified messages (seeds 150-152, 20 paired draws per cell)

Headline: either-fusion of the pair (asking reader, vMF) at n = 30 verified messages. Calibrators: plain (G4); trim k = the k largest scores of each reader's sample dropped; agree = the asking reader's top option must be an expert the outcome names. Modes: real = arithmetic's answer check (a quarter of the stream never-taught, as G4b); inj10 / inj20 = never-taught messages added as if verified so that they make 10 % / 20 % of the sample. Clean = the same random order without the wrong messages. Retention = contaminated found / clean found of the same calibrator.

## arithmetic (3 seeds; 120 known, 120 never-taught, 236 foreign per seed)

Verification: 0.558 of the known messages verified; never-taught wrongly vouched for 0.158; the named experts include the true owner for 0.889 of the verified known messages; the asking reader's top option is one of the named experts for 0.804 of them.

### arithmetic, mode real, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 29.8 | 0.091 | 0.051 | 0.248 | 0.036 | 0.171 | 0.689 |
| trim1 | 29.8 | 0.091 | 0.101 | 0.432 | 0.077 | 0.279 | 0.645 |
| trim2 | 29.8 | 0.091 | 0.145 | 0.596 | 0.114 | 0.416 | 0.697 |
| trim3 | 29.8 | 0.091 | 0.175 | 0.714 | 0.148 | 0.551 | 0.772 |
| trim6 | 29.8 | 0.091 | 0.276 | 0.922 | 0.242 | 0.838 | 0.909 |
| agree | 26.8 | 0.011 | 0.149 | 0.756 | 0.144 | 0.739 | 0.978 |

### arithmetic, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 29.9 | 0.104 | 0.050 | 0.230 | 0.037 | 0.141 | 0.612 |
| trim1 | 29.9 | 0.104 | 0.101 | 0.427 | 0.080 | 0.262 | 0.614 |
| trim2 | 29.9 | 0.104 | 0.146 | 0.602 | 0.114 | 0.397 | 0.659 |
| trim3 | 29.9 | 0.104 | 0.183 | 0.726 | 0.152 | 0.525 | 0.723 |
| trim6 | 29.9 | 0.104 | 0.286 | 0.925 | 0.253 | 0.836 | 0.904 |
| agree | 27.0 | 0.021 | 0.149 | 0.756 | 0.139 | 0.722 | 0.955 |

### arithmetic, mode inj20, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.206 | 0.050 | 0.225 | 0.033 | 0.121 | 0.538 |
| trim1 | 30.0 | 0.206 | 0.102 | 0.416 | 0.069 | 0.214 | 0.514 |
| trim2 | 30.0 | 0.206 | 0.148 | 0.598 | 0.100 | 0.295 | 0.493 |
| trim3 | 30.0 | 0.206 | 0.183 | 0.708 | 0.134 | 0.393 | 0.556 |
| trim6 | 30.0 | 0.206 | 0.284 | 0.929 | 0.218 | 0.689 | 0.741 |
| agree | 27.3 | 0.037 | 0.149 | 0.756 | 0.139 | 0.715 | 0.946 |

### arithmetic, retention under 10 % injected contamination as the sample grows

| n | plain | trim1 | trim3 | trim6 | agree |
|---|---|---|---|---|---|
| 30 | 0.612 | 0.614 | 0.723 | 0.904 | 0.955 |
| 40 | 0.608 | 0.717 | 0.682 | 0.813 | 0.956 |
| 60 | 0.603 | 0.693 | 0.684 | 0.802 | 0.956 |

## banking (3 seeds; 180 known, 300 never-taught, 150 foreign per seed)

Verification: 1.000 of the known messages verified; never-taught wrongly vouched for 0.000; the named experts include the true owner for 1.000 of the verified known messages; the asking reader's top option is one of the named experts for 0.894 of them.

### banking, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.101 | 0.060 | 0.676 | 0.022 | 0.362 | 0.535 |
| trim1 | 30.0 | 0.101 | 0.126 | 0.858 | 0.059 | 0.558 | 0.651 |
| trim2 | 30.0 | 0.101 | 0.175 | 0.916 | 0.100 | 0.697 | 0.761 |
| trim3 | 30.0 | 0.101 | 0.225 | 0.947 | 0.145 | 0.825 | 0.871 |
| trim6 | 30.0 | 0.101 | 0.374 | 0.980 | 0.280 | 0.950 | 0.969 |
| agree | 30.0 | 0.015 | 0.114 | 0.762 | 0.105 | 0.700 | 0.918 |

### banking, mode inj20, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.191 | 0.078 | 0.702 | 0.017 | 0.227 | 0.323 |
| trim1 | 30.0 | 0.191 | 0.138 | 0.855 | 0.043 | 0.399 | 0.467 |
| trim2 | 30.0 | 0.191 | 0.196 | 0.919 | 0.069 | 0.516 | 0.562 |
| trim3 | 30.0 | 0.191 | 0.251 | 0.953 | 0.107 | 0.672 | 0.705 |
| trim6 | 30.0 | 0.191 | 0.389 | 0.980 | 0.214 | 0.880 | 0.897 |
| agree | 30.0 | 0.032 | 0.120 | 0.767 | 0.097 | 0.624 | 0.814 |

### banking, the price of trimming on a clean sample (nothing wrongly verified)

| n | calibrator | theory (k+1)/(n+1) | asking reader | vMF | pair: known flagged | pair: found |
|---|---|---|---|---|---|---|
| 30 | plain | 0.032 | 0.032 | 0.039 | 0.068 | 0.700 |
| 30 | trim1 | 0.065 | 0.061 | 0.071 | 0.121 | 0.860 |
| 30 | trim2 | 0.097 | 0.095 | 0.104 | 0.176 | 0.912 |
| 30 | trim3 | 0.129 | 0.128 | 0.137 | 0.232 | 0.945 |
| 30 | trim6 | 0.226 | 0.221 | 0.220 | 0.371 | 0.978 |
| 40 | plain | 0.024 | 0.053 | 0.059 | 0.058 | 0.653 |
| 40 | trim1 | 0.049 | 0.081 | 0.089 | 0.154 | 0.896 |
| 40 | trim2 | 0.073 | 0.081 | 0.089 | 0.154 | 0.896 |
| 40 | trim3 | 0.098 | 0.106 | 0.107 | 0.190 | 0.928 |
| 40 | trim6 | 0.171 | 0.179 | 0.174 | 0.299 | 0.969 |
| 60 | plain | 0.016 | 0.057 | 0.051 | 0.066 | 0.717 |
| 60 | trim1 | 0.033 | 0.072 | 0.068 | 0.128 | 0.875 |
| 60 | trim2 | 0.049 | 0.072 | 0.068 | 0.128 | 0.875 |
| 60 | trim3 | 0.066 | 0.088 | 0.085 | 0.155 | 0.911 |
| 60 | trim6 | 0.115 | 0.139 | 0.128 | 0.232 | 0.956 |

### banking, retention under 10 % injected contamination as the sample grows

| n | plain | trim1 | trim3 | trim6 | agree |
|---|---|---|---|---|---|
| 30 | 0.535 | 0.651 | 0.871 | 0.969 | 0.918 |
| 40 | 0.482 | 0.678 | 0.775 | 0.947 | 0.895 |
| 60 | 0.480 | 0.620 | 0.685 | 0.881 | 0.893 |

## agnews (3 seeds; 180 known, 300 never-taught, 150 foreign per seed)

Verification: 1.000 of the known messages verified; never-taught wrongly vouched for 0.000; the named experts include the true owner for 1.000 of the verified known messages; the asking reader's top option is one of the named experts for 0.798 of them.

### agnews, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.101 | 0.044 | 0.416 | 0.026 | 0.267 | 0.643 |
| trim1 | 30.0 | 0.101 | 0.088 | 0.634 | 0.052 | 0.468 | 0.738 |
| trim2 | 30.0 | 0.101 | 0.138 | 0.751 | 0.086 | 0.626 | 0.834 |
| trim3 | 30.0 | 0.101 | 0.191 | 0.821 | 0.121 | 0.727 | 0.885 |
| trim6 | 30.0 | 0.101 | 0.342 | 0.917 | 0.264 | 0.881 | 0.961 |
| agree | 30.0 | 0.026 | 0.123 | 0.707 | 0.122 | 0.697 | 0.985 |

### agnews, mode inj20, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.191 | 0.051 | 0.470 | 0.022 | 0.238 | 0.506 |
| trim1 | 30.0 | 0.191 | 0.097 | 0.660 | 0.040 | 0.387 | 0.587 |
| trim2 | 30.0 | 0.191 | 0.146 | 0.757 | 0.061 | 0.529 | 0.699 |
| trim3 | 30.0 | 0.191 | 0.195 | 0.815 | 0.092 | 0.635 | 0.779 |
| trim6 | 30.0 | 0.191 | 0.351 | 0.915 | 0.195 | 0.829 | 0.906 |
| agree | 30.0 | 0.067 | 0.135 | 0.724 | 0.116 | 0.682 | 0.941 |

### agnews, the price of trimming on a clean sample (nothing wrongly verified)

| n | calibrator | theory (k+1)/(n+1) | asking reader | vMF | pair: known flagged | pair: found |
|---|---|---|---|---|---|---|
| 30 | plain | 0.032 | 0.031 | 0.029 | 0.055 | 0.458 |
| 30 | trim1 | 0.065 | 0.052 | 0.054 | 0.095 | 0.651 |
| 30 | trim2 | 0.097 | 0.091 | 0.089 | 0.155 | 0.771 |
| 30 | trim3 | 0.129 | 0.123 | 0.119 | 0.206 | 0.833 |
| 30 | trim6 | 0.226 | 0.201 | 0.230 | 0.361 | 0.924 |
| 40 | plain | 0.024 | 0.050 | 0.045 | 0.049 | 0.464 |
| 40 | trim1 | 0.049 | 0.072 | 0.069 | 0.123 | 0.730 |
| 40 | trim2 | 0.073 | 0.072 | 0.069 | 0.123 | 0.730 |
| 40 | trim3 | 0.098 | 0.095 | 0.091 | 0.159 | 0.791 |
| 40 | trim6 | 0.171 | 0.163 | 0.170 | 0.282 | 0.895 |
| 60 | plain | 0.016 | 0.047 | 0.039 | 0.053 | 0.501 |
| 60 | trim1 | 0.033 | 0.062 | 0.054 | 0.101 | 0.683 |
| 60 | trim2 | 0.049 | 0.062 | 0.054 | 0.101 | 0.683 |
| 60 | trim3 | 0.066 | 0.077 | 0.071 | 0.128 | 0.745 |
| 60 | trim6 | 0.115 | 0.118 | 0.113 | 0.196 | 0.839 |

### agnews, retention under 10 % injected contamination as the sample grows

| n | plain | trim1 | trim3 | trim6 | agree |
|---|---|---|---|---|---|
| 30 | 0.643 | 0.738 | 0.885 | 0.961 | 0.985 |
| 40 | 0.590 | 0.801 | 0.850 | 0.934 | 0.987 |
| 60 | 0.609 | 0.758 | 0.818 | 0.880 | 0.995 |

## newsgroups (3 seeds; 180 known, 300 never-taught, 150 foreign per seed)

Verification: 1.000 of the known messages verified; never-taught wrongly vouched for 0.000; the named experts include the true owner for 1.000 of the verified known messages; the asking reader's top option is one of the named experts for 0.596 of them.

### newsgroups, mode inj10, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.101 | 0.060 | 0.240 | 0.047 | 0.196 | 0.816 |
| trim1 | 30.0 | 0.101 | 0.121 | 0.377 | 0.092 | 0.315 | 0.835 |
| trim2 | 30.0 | 0.101 | 0.174 | 0.481 | 0.142 | 0.413 | 0.858 |
| trim3 | 30.0 | 0.101 | 0.221 | 0.564 | 0.185 | 0.498 | 0.882 |
| trim6 | 30.0 | 0.101 | 0.373 | 0.768 | 0.320 | 0.713 | 0.928 |
| agree | 30.0 | 0.009 | 0.357 | 0.772 | 0.355 | 0.769 | 0.997 |

### newsgroups, mode inj20, n = 30

| calibrator | n in sample | never-taught share | clean: known flagged | found | contaminated: known flagged | found | retention |
|---|---|---|---|---|---|---|---|
| plain | 30.0 | 0.191 | 0.064 | 0.258 | 0.039 | 0.181 | 0.703 |
| trim1 | 30.0 | 0.191 | 0.116 | 0.383 | 0.090 | 0.314 | 0.819 |
| trim2 | 30.0 | 0.191 | 0.171 | 0.487 | 0.128 | 0.406 | 0.834 |
| trim3 | 30.0 | 0.191 | 0.226 | 0.570 | 0.169 | 0.484 | 0.850 |
| trim6 | 30.0 | 0.191 | 0.364 | 0.761 | 0.301 | 0.682 | 0.896 |
| agree | 30.0 | 0.018 | 0.355 | 0.767 | 0.351 | 0.757 | 0.986 |

### newsgroups, the price of trimming on a clean sample (nothing wrongly verified)

| n | calibrator | theory (k+1)/(n+1) | asking reader | vMF | pair: known flagged | pair: found |
|---|---|---|---|---|---|---|
| 30 | plain | 0.032 | 0.034 | 0.035 | 0.067 | 0.262 |
| 30 | trim1 | 0.065 | 0.067 | 0.070 | 0.127 | 0.407 |
| 30 | trim2 | 0.097 | 0.099 | 0.110 | 0.191 | 0.512 |
| 30 | trim3 | 0.129 | 0.128 | 0.131 | 0.231 | 0.578 |
| 30 | trim6 | 0.226 | 0.211 | 0.237 | 0.369 | 0.766 |
| 40 | plain | 0.024 | 0.049 | 0.045 | 0.053 | 0.216 |
| 40 | trim1 | 0.049 | 0.074 | 0.069 | 0.134 | 0.415 |
| 40 | trim2 | 0.073 | 0.074 | 0.069 | 0.134 | 0.415 |
| 40 | trim3 | 0.098 | 0.098 | 0.098 | 0.179 | 0.496 |
| 40 | trim6 | 0.171 | 0.162 | 0.175 | 0.292 | 0.671 |
| 60 | plain | 0.016 | 0.048 | 0.053 | 0.065 | 0.265 |
| 60 | trim1 | 0.033 | 0.065 | 0.075 | 0.129 | 0.407 |
| 60 | trim2 | 0.049 | 0.065 | 0.075 | 0.129 | 0.407 |
| 60 | trim3 | 0.066 | 0.078 | 0.084 | 0.148 | 0.444 |
| 60 | trim6 | 0.115 | 0.129 | 0.136 | 0.236 | 0.585 |

### newsgroups, retention under 10 % injected contamination as the sample grows

| n | plain | trim1 | trim3 | trim6 | agree |
|---|---|---|---|---|---|
| 30 | 0.816 | 0.835 | 0.882 | 0.928 | 0.997 |
| 40 | 0.828 | 0.849 | 0.887 | 0.932 | 0.995 |
| 60 | 0.725 | 0.817 | 0.844 | 0.878 | 1.007 |

## Verdicts (rules fixed before the run)

- C11, trim1: clean known flagged 0.101, contaminated 0.077 (<= 0.1); found 0.279 against 0.432 clean, retention 0.645 (need >= 0.85). **FAIL**
- C11, trim2: clean known flagged 0.145, contaminated 0.114 (<= 0.1); found 0.416 against 0.596 clean, retention 0.697 (need >= 0.85). **FAIL**
- C11, trim3: clean known flagged 0.175, contaminated 0.148 (<= 0.1); found 0.551 against 0.714 clean, retention 0.772 (need >= 0.85). **FAIL**
- C11, agree: clean known flagged 0.149, contaminated 0.144 (<= 0.1); found 0.739 against 0.756 clean, retention 0.978 (need >= 0.85). **FAIL**
- C11 overall (plain keeps 0.689): **FAIL**
- C12 (single readers, clean, text domains): n=30 k=0: 0.033 vs 0.032; n=30 k=1: 0.063 vs 0.065; n=30 k=2: 0.098 vs 0.097; n=30 k=3: 0.127 vs 0.129; n=60 k=0: 0.049 vs 0.016; n=60 k=1: 0.066 vs 0.033; n=60 k=2: 0.066 vs 0.049; n=60 k=3: 0.081 vs 0.066; n=60 k=6: 0.127 vs 0.115. **FAIL** (tolerance 0.03)
- C13 overall (one calibrator in every domain under inj10): **FAIL**
