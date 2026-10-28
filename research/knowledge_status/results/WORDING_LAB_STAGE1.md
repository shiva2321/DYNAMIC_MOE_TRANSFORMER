# Mitigations lab, stage 1 (reader variants, tuning seeds 150-154)

### Reader variants, machine as it is

| reader | false alarm S4-S8 | stories S4 | noisy text S7 | swapped order S5 | taught wordings | never-taught skills found | macro F1 | Other called unseen | passed by the margin gate |
|---|---|---|---|---|---|---|---|---|---|
| plain | 0.256 | 0.624 | 0.450 | 0.183 | 0.037 | 0.903 | 0.866 | 0.275 | 0 |
| clean | 0.250 | 0.639 | 0.376 | 0.217 | 0.031 | 0.890 | 0.861 | 0.232 | 0 |
| p1 | 0.333 | 0.933 | 0.476 | 0.133 | 0.062 | 0.869 | 0.821 | 0.348 | 0 |
| p2 | 0.309 | 0.772 | 0.517 | 0.125 | 0.052 | 0.877 | 0.828 | 0.368 | 0 |
| clean+p1 | 0.251 | 0.730 | 0.322 | 0.133 | 0.053 | 0.863 | 0.843 | 0.314 | 0 |
| clean+p2 | 0.261 | 0.741 | 0.297 | 0.133 | 0.050 | 0.872 | 0.838 | 0.331 | 0 |

**Winner by the declared rule: `plain`**
