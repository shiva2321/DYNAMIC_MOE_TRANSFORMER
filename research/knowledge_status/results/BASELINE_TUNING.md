# Tuning of the standard detectors on `R5_R4_qwen_L26`, seeds 150-154 (not report seeds)

Criterion: mean AUROC of parts of untaught skills against parts of taught skills (no threshold). A setting within 0.002 of the best loses to the default. 'found' and 'false unseen' are at q = 0.05 with the shared held-out calibration and are shown for information only.

| detector | setting | seeds | mean AUROC | found | false unseen | chosen |
|---|---|---|---|---|---|---|
| knn_distance | k=1 | 5 | 0.929 | 0.910 | 0.340 | chosen |
| knn_distance | k=5 | 5 | 0.917 | 0.871 | 0.319 |  |
| knn_distance | k=10 (default) | 5 | 0.906 | 0.851 | 0.317 |  |
| knn_distance | k=20 | 5 | 0.855 | 0.755 | 0.300 |  |
| knn_distance | k=50 | 5 | 0.461 | 0.087 | 0.102 |  |
| vmf_conformal | max_protos=64 | 5 | 0.943 | 0.915 | 0.308 | chosen |
| vmf_conformal | max_protos=256 (default) | 5 | 0.941 | 0.910 | 0.308 |  |
| vmf_conformal | max_protos=1024 | 5 | 0.938 | 0.911 | 0.273 |  |
| mahalanobis | default (default) | 5 | 0.953 | 0.952 | 0.323 | chosen |
| relative_mahalanobis | default (default) | 5 | 0.406 | 0.081 | 0.017 | chosen |
| classifier_confidence | C=0.01 | 5 | 0.469 | 0.160 | 0.175 |  |
| classifier_confidence | C=0.1 | 5 | 0.598 | 0.397 | 0.245 |  |
| classifier_confidence | C=1.0 (default) | 5 | 0.652 | 0.507 | 0.287 | chosen |
| classifier_confidence | C=10.0 | 5 | 0.599 | 0.503 | 0.288 |  |
| energy | C=0.01 | 5 | 0.401 | 0.096 | 0.204 |  |
| energy | C=0.1 | 5 | 0.492 | 0.290 | 0.202 |  |
| energy | C=1.0 (default) | 5 | 0.498 | 0.296 | 0.172 |  |
| energy | C=10.0 | 5 | 0.565 | 0.291 | 0.082 | chosen |
| autoencoder_descriptor | epochs=100,hidden=8 | 5 | 0.863 | 0.627 | 0.285 |  |
| autoencoder_descriptor | epochs=300,hidden=8 | 5 | 0.866 | 0.652 | 0.292 |  |
| autoencoder_descriptor | epochs=100,hidden=16 | 5 | 0.867 | 0.653 | 0.287 |  |
| autoencoder_descriptor | epochs=300,hidden=16 | 5 | 0.870 | 0.659 | 0.285 |  |
| autoencoder_descriptor | epochs=100,hidden=32 | 5 | 0.870 | 0.669 | 0.307 |  |
| autoencoder_descriptor | epochs=300,hidden=32 (default) | 5 | 0.863 | 0.668 | 0.306 |  |
| autoencoder_descriptor | epochs=100,hidden=64 | 5 | 0.886 | 0.707 | 0.310 | chosen |
| autoencoder_descriptor | epochs=300,hidden=64 | 5 | 0.878 | 0.751 | 0.264 |  |

Chosen overrides (an empty dict keeps the default): `BASELINE_PARAMS = {"knn_distance": {"k": 1}, "vmf_conformal": {"max_protos": 64}, "mahalanobis": {}, "relative_mahalanobis": {}, "classifier_confidence": {}, "energy": {"C": 10.0}, "autoencoder_descriptor": {"hidden": 64, "epochs": 100}}`
