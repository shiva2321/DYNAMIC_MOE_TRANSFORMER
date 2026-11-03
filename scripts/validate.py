"""Human-readable stress-test trace for the SS-DM-MoE package.

Demonstrates sequential multimodal ingestion, forced vertical growth,
forced horizontal expert split, forced horizontal expert death, and a
full optimization step. Thresholds are forced explicitly via direct
buffer/attribute manipulation below -- nothing is hidden in tuned
constructor defaults.
"""

import torch
import torch.nn.functional as F

from dmoe.model import SelfStabilizingDynamicMorphicMoE


def run_validation_suite():
    print("=" * 80)
    print("SS-DM-MoE VALIDATION HARNESS")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Execution Target Device: {device}\n")

    d_model = 64
    model = SelfStabilizingDynamicMorphicMoE(
        d_model=d_model,
        num_heads=4,
        d_ff=128,
        initial_layers=2,
        initial_experts=3,
        theta_up=0.45,
        theta_down=0.08,
        min_layers=1,
        max_layers=6,
    ).to(device)

    print(f"Initial depth: {model.num_layers} layers")
    print(f"Initial experts per layer: {[l.moe.num_experts for l in model.layers]}\n")

    print("-" * 80)
    print("PHASE 1: Sequential multimodal ingestion")
    text_input = torch.randint(0, 1000, (4, 16), dtype=torch.long, device=device)
    out_text = model(text_input, output_head="token")
    print(
        f"  Text:     in={list(text_input.shape)} out={list(out_text['output'].shape)} "
        f"structural_error={out_text['structural_error'].item():.5f}"
    )

    image_input = torch.randn(4, 3, 32, 32, device=device)
    out_img = model(image_input, output_head="continuous")
    print(
        f"  Image:    in={list(image_input.shape)} out={list(out_img['output'].shape)} "
        f"structural_error={out_img['structural_error'].item():.5f}"
    )

    numeric_input = torch.randn(4, 24, 8, device=device)
    out_num = model(numeric_input, output_head="continuous")
    print(
        f"  Numeric:  in={list(numeric_input.shape)} out={list(out_num['output'].shape)} "
        f"structural_error={out_num['structural_error'].item():.5f}"
    )

    print("\n" + "-" * 80)
    print("PHASE 2: Forcing structural error above theta_up to trigger vertical growth")
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_(model.theta_up + 0.3)
    report = model.execute_graph_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Depth now: {model.num_layers} layers, experts: {report['experts_per_layer']}")

    print("\n" + "-" * 80)
    print("PHASE 3: Forcing routing entropy above tau_split to trigger expert mitosis")
    with torch.no_grad():
        model.manifold_corr.running_error_magnitude.fill_((model.theta_up + model.theta_down) * 0.5)
    model.layers[0].moe._last_entropy = torch.tensor(model.layers[0].moe.tau_split + 0.2)
    report = model.execute_graph_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Layer 0 experts now: {model.layers[0].moe.num_experts}")

    print("\n" + "-" * 80)
    print("PHASE 4: Forcing an expert's usage below tau_death to trigger expulsion")
    with torch.no_grad():
        model.layers[0].moe.expert_usage[0] = model.layers[0].moe.tau_death * 0.1
    model.layers[0].moe._last_entropy = torch.tensor(model.layers[0].moe.tau_split * 0.1)
    report = model.execute_graph_mutation()
    for log in report["mutation_logs"]:
        print(f"  * {log}")
    print(f"  Layer 0 experts now: {model.layers[0].moe.num_experts}")

    print("\n" + "-" * 80)
    print("PHASE 5: Full optimization step on the mutated architecture")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    forward_res = model(text_input, output_head="token")
    dummy_targets = torch.randint(0, 1000, (4, 16), device=device)
    task_loss = F.cross_entropy(
        forward_res["output"].view(-1, forward_res["output"].size(-1)),
        dummy_targets.view(-1),
    )
    total_loss = task_loss + 0.1 * forward_res["structural_error"]
    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    print(f"  Total loss: {total_loss.item():.4f}  Grad norm: {grad_norm.item():.4f}")
    print("\n" + "=" * 80)
    print("VALIDATION HARNESS COMPLETE")
    print("=" * 80)


run = run_validation_suite


if __name__ == "__main__":
    run_validation_suite()
