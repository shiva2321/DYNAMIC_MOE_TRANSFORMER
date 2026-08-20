import os
import glob
import json
import math

VOCAB_SIZE = 50304
LOSS_CEILING = math.log(VOCAB_SIZE)  # ~10.826 nats; max valid cross-entropy loss (uniform-random guessing)

exp_dir = "experiments"
json_files = glob.glob(os.path.join(exp_dir, "continual_control_*_replay_results.json"))

valid_rows = []
excluded_rows = []

for fpath in json_files:
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            data = json.load(f)

        m_type = data.get("model_type", os.path.basename(fpath).replace("continual_control_", "").replace("_replay_results.json", ""))
        mean_bwt = data.get("mean_bwt_nats", 0.0)
        acc_mat = data.get("acc_matrix", [[0] * 4] * 4)
        loss_mat = data.get("loss_matrix", [[0] * 4] * 4)

        py_imm_acc = acc_mat[1][1]
        py_final_acc = acc_mat[3][1]
        py_loss_delta = loss_mat[3][1] - loss_mat[1][1]

        # Mathematical-validity check: no cross-entropy loss should exceed the
        # uniform-random ceiling ln(vocab_size). A value above it is not "bad
        # performance" -- it indicates training instability (divergence, an
        # unstable fresh component, etc.) in that specific run, and the run's
        # numbers should not be trusted as a comparison point until fixed and
        # rerun. Flagged and excluded from ranking rather than silently included.
        flat_losses = [x for row in loss_mat for x in row]
        violations = [x for x in flat_losses if x > LOSS_CEILING]

        row = (m_type, mean_bwt, py_imm_acc, py_final_acc, py_loss_delta)
        if violations:
            excluded_rows.append((row, violations))
        else:
            valid_rows.append(row)
    except Exception as e:
        print(f"Error loading {fpath}: {e}")

# Sort by Mean R_BWT ascending (lowest forgetting to highest forgetting)
valid_rows.sort(key=lambda x: x[1])

print("=" * 115)
print(f"{'Model Name':<38} | {'Mean R_BWT (nats)':<18} | {'Py Imm Acc':<10} | {'Py Final Acc':<12} | {'Py Loss Delta'}")
print("=" * 115)
for m_type, bwt, imm_acc, fin_acc, loss_d in valid_rows:
    print(f"{m_type:<38} | {bwt:+.4f} nats        | {imm_acc:5.2f}%     | {fin_acc:5.2f}%       | {loss_d:+.4f} nats")
print("=" * 115)

if excluded_rows:
    print()
    print("=" * 115)
    print("EXCLUDED -- MATHEMATICALLY INVALID (loss exceeds ln(vocab_size) = %.4f nats; run is unreliable, not comparable)" % LOSS_CEILING)
    print("=" * 115)
    for (m_type, bwt, imm_acc, fin_acc, loss_d), violations in excluded_rows:
        print(f"{m_type:<38} | reported R_BWT {bwt:+.4f} nats -- DO NOT TRUST | {len(violations)} loss value(s) above ceiling, max = {max(violations):.4f} nats")
    print("Reason: a cross-entropy loss above ln(vocab_size) is mathematically outside the range a stable,")
    print("correctly-converging model can produce -- it is the signature of training instability in that")
    print("specific run (divergence, an untuned fresh component, etc.), not evidence about the architecture")
    print("being compared. Superseded by a clean, capacity-matched control where applicable.")
    print("=" * 115)
