# Lab G4c, post-hoc: one curve of false flags against found? (seeds 150-152; reads the cached cells; the prediction was written before this was computed)

A calibrator only chooses a cut. For ONE reader every calibrator and both arms are judged on the same evaluation messages, so all their (known flagged, found) points belong to one curve. The clean curve is drawn through the clean points of every calibrator, mode and sample size (corners (0,0) and (1,1) added, linear in between). 'Lost at matched false flags' = what the clean curve finds at the point's known-flagged rate minus what the point finds. For the PAIR the curve is only approximate (the fused score depends on the sample), and agree is left out of it.

## arithmetic

### arithmetic, mode real, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.036 | 0.171 | -0.007 | -0.035 | 0.009 | 0.689 | 1.043 |
| trim1 | 0.077 | 0.279 | 0.056 | -0.005 | 0.002 | 0.645 | 0.832 |
| trim2 | 0.114 | 0.416 | 0.062 | -0.010 | 0.001 | 0.697 | 0.870 |
| trim3 | 0.148 | 0.551 | 0.053 | -0.020 | 0.007 | 0.772 | 0.912 |
| trim6 | 0.242 | 0.838 | 0.044 | -0.015 | -0.003 | 0.909 | 0.950 |
| agree | 0.144 | 0.739 | -0.145 | -0.033 | 0.000 | 0.978 | n/a |

### arithmetic, mode inj10, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.037 | 0.141 | 0.026 | -0.033 | -0.002 | 0.612 | 0.847 |
| trim1 | 0.080 | 0.262 | 0.084 | -0.006 | 0.001 | 0.614 | 0.758 |
| trim2 | 0.114 | 0.397 | 0.079 | -0.022 | 0.000 | 0.659 | 0.834 |
| trim3 | 0.152 | 0.525 | 0.096 | -0.011 | -0.003 | 0.723 | 0.846 |
| trim6 | 0.253 | 0.836 | 0.069 | -0.003 | 0.014 | 0.904 | 0.923 |
| agree | 0.139 | 0.722 | -0.148 | -0.024 | 0.000 | 0.955 | n/a |

### arithmetic, mode inj20, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.033 | 0.121 | 0.029 | -0.023 | 0.006 | 0.538 | 0.805 |
| trim1 | 0.069 | 0.214 | 0.095 | -0.033 | 0.002 | 0.514 | 0.694 |
| trim2 | 0.100 | 0.295 | 0.129 | -0.010 | -0.010 | 0.493 | 0.695 |
| trim3 | 0.134 | 0.393 | 0.162 | 0.002 | -0.003 | 0.556 | 0.708 |
| trim6 | 0.218 | 0.689 | 0.131 | -0.025 | 0.001 | 0.741 | 0.840 |
| agree | 0.139 | 0.715 | -0.140 | -0.023 | -0.001 | 0.946 | n/a |

agree against the pair's trim curve (arithmetic, clean): known flagged 0.149, found 0.756; the curve finds 0.605 at that rate, so agree is ABOVE it by 0.150. The asking reader agrees with the named experts on 0.804 of the verified known messages (1 minus that: 0.196); under agree the pair flags 0.149 and the asking reader alone 0.128.

## banking

### banking, mode inj10, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.022 | 0.362 | -0.079 | -0.019 | -0.174 | 0.535 | 1.282 |
| trim1 | 0.059 | 0.558 | 0.107 | 0.015 | -0.127 | 0.651 | 0.839 |
| trim2 | 0.100 | 0.697 | 0.111 | 0.018 | 0.029 | 0.761 | 0.863 |
| trim3 | 0.145 | 0.825 | 0.080 | -0.002 | 0.037 | 0.871 | 0.912 |
| trim6 | 0.280 | 0.950 | 0.017 | 0.007 | 0.019 | 0.969 | 0.983 |
| agree | 0.105 | 0.700 | 0.120 | -0.001 | 0.065 | 0.918 | n/a |

### banking, mode inj20, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.017 | 0.227 | -0.007 | -0.021 | -0.097 | 0.323 | 1.034 |
| trim1 | 0.043 | 0.399 | 0.137 | 0.010 | -0.134 | 0.467 | 0.745 |
| trim2 | 0.069 | 0.516 | 0.207 | 0.029 | -0.162 | 0.562 | 0.713 |
| trim3 | 0.107 | 0.672 | 0.152 | 0.030 | -0.073 | 0.705 | 0.816 |
| trim6 | 0.214 | 0.880 | 0.062 | -0.003 | 0.058 | 0.897 | 0.934 |
| agree | 0.097 | 0.624 | 0.174 | -0.001 | 0.039 | 0.814 | n/a |

agree against the pair's trim curve (banking, clean): known flagged 0.114, found 0.762; the curve finds 0.841 at that rate, so agree is BELOW it by 0.079. The asking reader agrees with the named experts on 0.894 of the verified known messages (1 minus that: 0.106); under agree the pair flags 0.114 and the asking reader alone 0.078.

## agnews

### agnews, mode inj10, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.026 | 0.267 | -0.014 | 0.017 | -0.021 | 0.643 | 1.057 |
| trim1 | 0.052 | 0.468 | 0.020 | 0.015 | -0.014 | 0.738 | 0.958 |
| trim2 | 0.086 | 0.626 | -0.000 | 0.009 | 0.033 | 0.834 | 1.001 |
| trim3 | 0.121 | 0.727 | -0.007 | 0.012 | 0.014 | 0.885 | 1.009 |
| trim6 | 0.264 | 0.881 | 0.001 | 0.005 | 0.004 | 0.961 | 0.999 |
| agree | 0.122 | 0.697 | 0.028 | 0.000 | -0.015 | 0.985 | n/a |

### agnews, mode inj20, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.022 | 0.238 | -0.026 | 0.009 | -0.031 | 0.506 | 1.121 |
| trim1 | 0.040 | 0.387 | -0.009 | 0.020 | -0.038 | 0.587 | 1.024 |
| trim2 | 0.061 | 0.529 | 0.008 | 0.016 | -0.041 | 0.699 | 0.985 |
| trim3 | 0.092 | 0.635 | 0.009 | 0.015 | 0.032 | 0.779 | 0.986 |
| trim6 | 0.195 | 0.829 | -0.008 | 0.010 | 0.019 | 0.906 | 1.010 |
| agree | 0.116 | 0.682 | 0.036 | -0.006 | 0.008 | 0.941 | n/a |

agree against the pair's trim curve (agnews, clean): known flagged 0.123, found 0.707; the curve finds 0.732 at that rate, so agree is BELOW it by 0.025. The asking reader agrees with the named experts on 0.798 of the verified known messages (1 minus that: 0.202); under agree the pair flags 0.123 and the asking reader alone 0.104.

## newsgroups

### newsgroups, mode inj10, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.047 | 0.196 | 0.020 | -0.009 | 0.005 | 0.816 | 0.905 |
| trim1 | 0.092 | 0.315 | 0.011 | -0.000 | 0.012 | 0.835 | 0.967 |
| trim2 | 0.142 | 0.413 | 0.017 | -0.001 | 0.014 | 0.858 | 0.960 |
| trim3 | 0.185 | 0.498 | 0.003 | 0.005 | 0.015 | 0.882 | 0.993 |
| trim6 | 0.320 | 0.713 | -0.007 | 0.009 | -0.004 | 0.928 | 1.010 |
| agree | 0.355 | 0.769 | -0.020 | -0.002 | 0.028 | 0.997 | n/a |

### newsgroups, mode inj20, n = 30: contaminated points against the clean curve

| calibrator | pair: known flagged | found | lost at matched | asking reader: lost | vMF: lost | retention as defined in G4c | retention at matched false flags (pair) |
|---|---|---|---|---|---|---|---|
| plain | 0.039 | 0.181 | 0.003 | -0.005 | -0.017 | 0.703 | 0.984 |
| trim1 | 0.090 | 0.314 | 0.008 | -0.000 | 0.015 | 0.819 | 0.974 |
| trim2 | 0.128 | 0.406 | 0.004 | -0.004 | -0.018 | 0.834 | 0.990 |
| trim3 | 0.169 | 0.484 | -0.003 | -0.008 | 0.003 | 0.850 | 1.007 |
| trim6 | 0.301 | 0.682 | 0.001 | -0.004 | 0.008 | 0.896 | 0.998 |
| agree | 0.351 | 0.757 | -0.013 | 0.006 | 0.009 | 0.986 | n/a |

agree against the pair's trim curve (newsgroups, clean): known flagged 0.357, found 0.772; the curve finds 0.752 at that rate, so agree is ABOVE it by 0.020. The asking reader agrees with the named experts on 0.596 of the verified known messages (1 minus that: 0.404); under agree the pair flags 0.357 and the asking reader alone 0.336.

## The corrected flag-rate formula against the clean text-domain single readers

| n | k | observed (mean of the two readers over the three text domains) | (k+1)/(n+1), declared | corrected formula |
|---|---|---|---|---|
| 30 | 0 | 0.033 | 0.032 | 0.032 |
| 30 | 1 | 0.063 | 0.065 | 0.065 |
| 30 | 2 | 0.098 | 0.097 | 0.097 |
| 30 | 3 | 0.127 | 0.129 | 0.129 |
| 40 | 0 | 0.050 | 0.024 | 0.049 |
| 40 | 1 | 0.076 | 0.049 | 0.073 |
| 40 | 2 | 0.076 | 0.073 | 0.073 |
| 40 | 3 | 0.099 | 0.098 | 0.098 |
| 40 | 6 | 0.170 | 0.171 | 0.171 |
| 60 | 0 | 0.049 | 0.016 | 0.049 |
| 60 | 1 | 0.066 | 0.033 | 0.066 |
| 60 | 2 | 0.066 | 0.049 | 0.066 |
| 60 | 3 | 0.081 | 0.066 | 0.082 |
| 60 | 6 | 0.127 | 0.115 | 0.131 |

Largest difference between observed and the corrected formula: 0.004.

## Largest distance of a contaminated single-reader point from the clean curve (prediction 1: at most 0.05)

- arithmetic/real/ask_exemplars: 0.035
- arithmetic/real/vmf_conformal: 0.009
- arithmetic/inj10/ask_exemplars: 0.033
- arithmetic/inj10/vmf_conformal: 0.014
- arithmetic/inj20/ask_exemplars: 0.033
- arithmetic/inj20/vmf_conformal: 0.010
- banking/inj10/ask_exemplars: 0.019
- banking/inj10/vmf_conformal: 0.174  (over 0.05)
- banking/inj20/ask_exemplars: 0.030
- banking/inj20/vmf_conformal: 0.162  (over 0.05)
- agnews/inj10/ask_exemplars: 0.017
- agnews/inj10/vmf_conformal: 0.033
- agnews/inj20/ask_exemplars: 0.020
- agnews/inj20/vmf_conformal: 0.041
- newsgroups/inj10/ask_exemplars: 0.009
- newsgroups/inj10/vmf_conformal: 0.028
- newsgroups/inj20/ask_exemplars: 0.008
- newsgroups/inj20/vmf_conformal: 0.018
