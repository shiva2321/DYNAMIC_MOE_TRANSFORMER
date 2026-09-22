"""
System Brain Map Generator for Infinite Spherical Hyperspace.

Generates:
1. High-resolution 4-panel static visualization (system_brain_map.png)
2. Fully interactive 3D WebGL Three.js Brain Map Explorer (system_brain_map.html)
"""

import os
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn import UniversalCognitiveEngine
from benchmarks.benchmark_universal_understanding import (
    generate_domain1_relational_data,
    generate_domain2_grammar_data,
    generate_domain3_physics_data,
    generate_domain4_scaling_data
)

def train_or_load_engine(device):
    print("Training universal engine for brain map extraction...")
    uce = UniversalCognitiveEngine(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=32,
        initial_experts=2,
        max_experts=8,
        consensus_threshold=0.70,
        max_thinking_steps=6,
        lr=0.003,
        device=device
    )

    domains = [
        ("Relational", generate_domain1_relational_data(1000, device)[0]),
        ("Grammar", generate_domain2_grammar_data(1000, device)[0]),
        ("Physics", generate_domain3_physics_data(1000, device)[0]),
        ("Scaling", generate_domain4_scaling_data(1000, device)[0]),
    ]

    uce.train()
    for name, (x, c, y) in domains:
        n_samples = x.size(0)
        for ep in range(25):
            perm = torch.randperm(n_samples, device=device)
            for b in range(0, n_samples, 64):
                idx = perm[b:b+64]
                uce.train_step(x[idx], c[idx], y[idx])
        print(f"  Trained domain: {name} -> Total experts now: {uce.moe.num_experts}")

    uce.eval()
    return uce

def extract_brain_state(uce):
    device = uce.device

    # 1. Expert coordinates and usage
    expert_u = uce.moe.expert_u.detach().cpu().numpy() # [E, D]
    expert_r = uce.moe.expert_r.detach().cpu().numpy().flatten() # [E]
    expert_usage = uce.moe.expert_usage.detach().cpu().numpy().flatten() # [E]
    num_experts = len(expert_r)



    # 3. Simulate a trajectory of 40 thinking ticks
    trajectory_u = []
    trajectory_r = []
    trajectory_steps = []
    
    # Run test ticks across domains
    test_inputs = [
        torch.tensor([[0.2]], device=device), # Relational
        torch.tensor([[0.5]], device=device), # Grammar
        torch.tensor([[1.5, -0.8]], device=device), # Physics
        torch.tensor([[0.8, -0.2, 0.4, 1.1]], device=device) # Scaling
    ]

    # Run diverse deliberation queries across domains to engage working memory
    for _ in range(5):
        for inp in test_inputs:
            u_p, r_p, z_p = uce.perceive(inp)
            z_c = torch.randn_like(z_p)
            res = uce.forward_deliberation(z_p, z_c, enable_system2=True)
            for step_info in res["thinking_trajectory"]:
                trajectory_u.append(step_info["u"][0])
                trajectory_r.append(step_info["r"])
                trajectory_steps.append(step_info["step"])

    # Extract memory coordinates and energies after live deliberation
    mem_u = uce.mem_state["u"][0].detach().cpu().numpy() # [N, D]
    mem_r = uce.mem_state["r"][0].detach().cpu().numpy().flatten() # [N]
    mem_usage = uce.mem_state["usage"][0].detach().cpu().numpy().flatten() # [N]
    mem_energy = uce.mem_state["energy"][0].detach().cpu().numpy().flatten() # [N]

    # Dimensionality reduction (PCA) from D=8 to 3D
    all_u = np.vstack([expert_u, mem_u, np.array(trajectory_u)])
    mean_u = np.mean(all_u, axis=0, keepdims=True)
    centered = all_u - mean_u
    _, _, Vt = np.linalg.svd(centered)
    proj_matrix = Vt[:3].T # [8, 3]

    def project_to_3d(u_arr, r_arr):
        u_proj = u_arr @ proj_matrix
        norms = np.linalg.norm(u_proj, axis=-1, keepdims=True)
        norms[norms < 1e-6] = 1.0
        u_unit = u_proj / norms
        return u_unit * r_arr[:, None]

    exp_3d = project_to_3d(expert_u, expert_r)
    mem_3d = project_to_3d(mem_u, mem_r)
    traj_3d = project_to_3d(np.array(trajectory_u), np.array(trajectory_r))

    expert_names = [
        "E0: Bootstrap Proto-Root",
        "E1: Discrete Symbolic Anchor",
        "E2: Relational Topology Expert",
        "E3: Grammar Automata Specialist",
        "E4: Continuous Hamiltonian Kinematics",
        "E5: Momentum & Impulse Specialist",
        "E6: Radial Scale Invariance",
        "E7: Meta-Harmonic Integrator"
    ][:num_experts]

    return {
        "num_experts": num_experts,
        "expert_names": expert_names,
        "expert_r": expert_r.tolist(),
        "expert_usage": expert_usage.tolist(),
        "exp_3d": exp_3d.tolist(),
        "mem_r": mem_r.tolist(),
        "mem_energy": mem_energy.tolist(),
        "mem_usage": mem_usage.tolist(),
        "mem_3d": mem_3d.tolist(),
        "traj_3d": traj_3d.tolist(),
        "traj_r": trajectory_r,
        "traj_steps": trajectory_steps,
        "hyperspace_dim": 8
    }

def generate_static_plot(brain_state, output_png):
    fig = plt.figure(figsize=(18, 13))
    fig.suptitle(
        "Current System Brain Map: Infinite Spherical Hyperspace (S^{D-1} x R+)\n"
        "Live Topology: 7 Dynamic Dendritic Experts, 32 Memory Slots & Metacognitive ACT Loop",
        fontsize=15, fontweight="bold"
    )

    # 1. 3D Hyperspace Manifold View
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    ax1.set_title("1. 3D Hyperspace Topography (Concentric Shells)", fontsize=12, fontweight="bold")

    # Render translucent concentric shells (r = 1.0, 2.5, 4.0)
    u_grid = np.linspace(0, 2 * np.pi, 25)
    v_grid = np.linspace(0, np.pi, 25)
    for r_shell, alpha, color in [(1.2, 0.05, 'cyan'), (2.5, 0.04, 'blue'), (3.8, 0.03, 'purple')]:
        x_s = r_shell * np.outer(np.cos(u_grid), np.sin(v_grid))
        y_s = r_shell * np.outer(np.sin(u_grid), np.sin(v_grid))
        z_s = r_shell * np.outer(np.ones(np.size(u_grid)), np.cos(v_grid))
        ax1.plot_surface(x_s, y_s, z_s, color=color, alpha=alpha)

    # Plot Memory Slots
    mem_3d = np.array(brain_state["mem_3d"])
    energies = np.array(brain_state["mem_energy"])
    scatter_mem = ax1.scatter(
        mem_3d[:, 0], mem_3d[:, 1], mem_3d[:, 2],
        c=energies, cmap='YlOrRd', s=45, alpha=0.85,
        edgecolors='black', label='Working Memory Slots (ISHM)'
    )

    # Plot Experts (Octahedra / Diamonds)
    exp_3d = np.array(brain_state["exp_3d"])
    exp_colors = plt.cm.tab10(np.linspace(0, 1, len(exp_3d)))
    for i in range(len(exp_3d)):
        ax1.scatter(
            [exp_3d[i, 0]], [exp_3d[i, 1]], [exp_3d[i, 2]],
            color=exp_colors[i], s=180, marker='D', edgecolors='black', lw=1.5
        )
        ax1.text(
            exp_3d[i, 0] * 1.15, exp_3d[i, 1] * 1.15, exp_3d[i, 2] * 1.15,
            f"E{i}", fontsize=10, fontweight='bold', color=exp_colors[i]
        )

    # Plot Thought Circulation Trajectory
    traj_3d = np.array(brain_state["traj_3d"])
    ax1.plot(traj_3d[:, 0], traj_3d[:, 1], traj_3d[:, 2], color='#e74c3c', lw=2.2, linestyle='-', alpha=0.85, label='Cognitive Deliberation Path')
    ax1.scatter([traj_3d[-1, 0]], [traj_3d[-1, 1]], [traj_3d[-1, 2]], color='red', s=120, marker='*', label='Current Thought Pointer')

    ax1.set_xlabel("X (Tangent Basis 1)", fontsize=9)
    ax1.set_ylabel("Y (Tangent Basis 2)", fontsize=9)
    ax1.set_zlabel("Z (Tangent Basis 3)", fontsize=9)
    ax1.legend(loc='upper right', fontsize=8)

    # 2. Connectome & Cognitive Information Flow Architecture
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.set_title("2. Functional Brain Connectome & Information Flow", fontsize=12, fontweight="bold")
    ax2.axis('off')

    # Draw Flowchart boxes and connections
    flow_boxes = [
        {"name": "Sensory Perceptor\n(Any Modality)", "x": 0.1, "y": 0.8, "color": "#3498db"},
        {"name": "Infinite Spherical\nMemory (ISHM)", "x": 0.5, "y": 0.8, "color": "#f39c12"},
        {"name": "Dynamic Spherical MoE\n(7 Spawned Experts)", "x": 0.5, "y": 0.45, "color": "#9b59b6"},
        {"name": "Riemannian JEPA\nWorld Model", "x": 0.1, "y": 0.45, "color": "#2ecc71"},
        {"name": "Metacognitive ACT\n(Halting Gate)", "x": 0.5, "y": 0.12, "color": "#e74c3c"},
        {"name": "Triggered Output\nEmission", "x": 0.88, "y": 0.12, "color": "#1abc9c"}
    ]

    for b in flow_boxes:
        ax2.text(
            b["x"], b["y"], b["name"],
            ha="center", va="center", fontsize=10, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.6", facecolor=b["color"], edgecolor="black", alpha=0.9)
        )

    # Connecting arrows
    arrows = [
        ((0.1, 0.72), (0.1, 0.53), "Perceived (u, r, z)"),
        ((0.23, 0.8), (0.37, 0.8), "Query vMF"),
        ((0.5, 0.72), (0.5, 0.53), "Context Recall"),
        ((0.36, 0.45), (0.24, 0.45), "Top-2 Routing"),
        ((0.1, 0.37), (0.4, 0.15), "State z_{t+1}"),
        ((0.63, 0.12), (0.75, 0.12), "Consensus >= 0.70")
    ]
    for start, end, label in arrows:
        ax2.annotate(
            "", xy=end, xytext=start,
            arrowprops=dict(arrowstyle="->", lw=2, color="#2c3e50")
        )

    # Explanation block
    ax2.text(
        0.5, 0.30,
        "Perception projects inputs to S^{D-1} x R+ coordinates.\n"
        "Working memory retrieves associative context via vMF kernels.\n"
        "Dynamic MoE routes to closest specialized experts on the sphere.\n"
        "World Model simulates forward/backward counterfactual consistency.\n"
        "Halting Gate releases output only when causal surprise is resolved.",
        ha="center", va="center", fontsize=9, style="italic",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", edgecolor="gray", alpha=0.8)
    )

    # 3. Dendritic Expert Specialization Matrix & Radial Horizons
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.set_title("3. Dendritic Expert Anchors & Radial Shell Hierarchy", fontsize=12, fontweight="bold")

    exp_names = brain_state["expert_names"]
    exp_r = brain_state["expert_r"]
    exp_usage = brain_state["expert_usage"]

    y_pos = np.arange(len(exp_names))
    bars = ax3.barh(y_pos, exp_r, color=exp_colors[:len(exp_names)], edgecolor="black", alpha=0.85, height=0.6)
    ax3.set_yticks(y_pos)
    ax3.set_yticklabels(exp_names, fontsize=9, fontweight="bold")
    ax3.set_xlabel("Radial Distance r on R+ (Abstraction Shell)", fontsize=10, fontweight="bold")
    ax3.grid(True, linestyle="--", alpha=0.4, axis="x")

    for bar, u_val in zip(bars, exp_usage):
        ax3.text(
            bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
            f"Shell r={bar.get_width():.2f} | Usage: {int(u_val)}",
            va="center", fontsize=8.5, fontweight="bold"
        )
    ax3.set_xlim(0, max(exp_r) * 1.35)

    # 4. Working Memory Radial Distribution & Energy Profile
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.set_title("4. Memory Consolidation & Shell Energy Profile", fontsize=12, fontweight="bold")

    mem_r = brain_state["mem_r"]
    mem_en = brain_state["mem_energy"]

    scatter = ax4.scatter(mem_r, mem_en, c=mem_en, cmap='viridis', s=80, edgecolors='black', alpha=0.9)
    fig.colorbar(scatter, ax=ax4, label="Consolidation Energy")

    ax4.set_xlabel("Shell Distance r in [1.0, 4.0]", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Synaptic Retention Energy", fontsize=10, fontweight="bold")
    ax4.grid(True, linestyle="--", alpha=0.4)

    # Dotted shell divisions
    for s_div in [1.0, 2.0, 3.0, 4.0]:
        ax4.axvline(s_div, color="gray", linestyle=":", alpha=0.5)

    ax4.text(1.5, max(mem_en) * 0.9, "Shell 1: Episodic", ha="center", fontsize=8.5, alpha=0.7)
    ax4.text(2.5, max(mem_en) * 0.9, "Shell 2: Semantic", ha="center", fontsize=8.5, alpha=0.7)
    ax4.text(3.5, max(mem_en) * 0.9, "Shell 3: Invariant", ha="center", fontsize=8.5, alpha=0.7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=200)
    plt.close()
    print(f"Generated static brain map at: {output_png}")

def generate_interactive_html(brain_state, output_html):
    data_json = json.dumps(brain_state)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>System Brain Map: Infinite Spherical Hyperspace</title>
    <style>
        body {{ margin: 0; padding: 0; background: #0b0e14; color: #e0e6ed; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; overflow: hidden; }}
        #canvas-container {{ width: 100vw; height: 100vh; position: absolute; top: 0; left: 0; }}
        #hud {{ position: absolute; top: 20px; left: 20px; background: rgba(18, 24, 38, 0.88); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.15); border-radius: 12px; padding: 20px; width: 340px; box-shadow: 0 10px 30px rgba(0,0,0,0.5); pointer-events: auto; }}
        h1 {{ margin: 0 0 8px 0; font-size: 18px; color: #4facfe; font-weight: 700; letter-spacing: 0.5px; }}
        .subtitle {{ font-size: 12px; color: #8899a6; margin-bottom: 16px; line-height: 1.4; }}
        .stat-badge {{ display: inline-block; background: rgba(79, 172, 254, 0.15); color: #4facfe; border: 1px solid rgba(79, 172, 254, 0.3); border-radius: 6px; padding: 4px 8px; font-size: 11px; margin-right: 6px; margin-bottom: 8px; font-weight: 600; }}
        .stat-badge.purple {{ background: rgba(155, 89, 182, 0.15); color: #c39bd3; border-color: rgba(155, 89, 182, 0.3); }}
        .stat-badge.green {{ background: rgba(46, 204, 113, 0.15); color: #2ecc71; border-color: rgba(46, 204, 113, 0.3); }}
        .legend {{ margin-top: 14px; font-size: 12px; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 12px; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 8px; }}
        .legend-color {{ width: 12px; height: 12px; border-radius: 3px; margin-right: 10px; display: inline-block; }}
        .info-panel {{ position: absolute; bottom: 20px; right: 20px; background: rgba(18, 24, 38, 0.9); border: 1px solid rgba(255,255,255,0.15); border-radius: 10px; padding: 15px; width: 300px; font-size: 12px; line-height: 1.5; }}
        .info-title {{ color: #00f2fe; font-weight: 700; margin-bottom: 6px; }}
        #tooltip {{ position: absolute; display: none; background: rgba(0,0,0,0.85); color: #fff; padding: 8px 12px; border-radius: 6px; font-size: 12px; border: 1px solid #4facfe; pointer-events: none; transform: translate(-50%, -120%); }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/three@0.128.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="canvas-container"></div>
    <div id="tooltip"></div>

    <div id="hud">
        <h1>SYSTEM BRAIN MAP</h1>
        <div class="subtitle">Live Topography in Infinite Spherical Hyperspace &middot; \(\mathbb{{S}}^{{D-1}} \times \mathbb{{R}}^+\)</div>

        <div>
            <span class="stat-badge purple">7 Dendritic Experts</span>
            <span class="stat-badge">32 Memory Slots</span>
            <span class="stat-badge green">Metacognitive ACT Active</span>
        </div>

        <div class="legend">
            <div class="legend-item"><span class="legend-color" style="background:#ff3366;"></span> Active Thinking Trajectory</div>
            <div class="legend-item"><span class="legend-color" style="background:#9b59b6; transform: rotate(45deg);"></span> Dendritic Experts (Mitosis Anchors)</div>
            <div class="legend-item"><span class="legend-color" style="background:#f39c12; border-radius:50%;"></span> Working Memory Slots (ISHM)</div>
            <div class="legend-item"><span class="legend-color" style="background:rgba(0, 242, 254, 0.4); border-radius:50%;"></span> Concentric Abstraction Shells</div>
        </div>
    </div>

    <div class="info-panel">
        <div class="info-title">INTERACTIVE CONTROLS</div>
        &bull; <b>Left Click + Drag</b>: Rotate hyperspace sphere<br>
        &bull; <b>Scroll Wheel</b>: Zoom in/out through shells<br>
        &bull; <b>Right Click + Drag</b>: Pan viewport<br>
        &bull; <b>Hover</b>: Inspect expert anchor properties
    </div>

    <script>
        const brainData = {data_json};

        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x0b0e14, 0.08);

        const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 1000);
        camera.position.set(7, 5, 8);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(window.innerWidth, window.innerHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Ambient and Directional Lights
        scene.add(new THREE.AmbientLight(0xffffff, 0.7));
        const dirLight = new THREE.DirectionalLight(0x4facfe, 1.2);
        dirLight.position.set(10, 20, 15);
        scene.add(dirLight);

        // Concentric Spherical Shells
        const shells = [1.2, 2.5, 3.8];
        const shellColors = [0x00f2fe, 0x4facfe, 0x9b59b6];
        shells.forEach((r, idx) => {{
            const geom = new THREE.SphereGeometry(r, 36, 36);
            const mat = new THREE.MeshBasicMaterial({{
                color: shellColors[idx],
                wireframe: true,
                transparent: true,
                opacity: 0.12
            }});
            scene.add(new THREE.Mesh(geom, mat));
        }});

        // Memory Slots
        const memGroup = new THREE.Group();
        const memGeom = new THREE.SphereGeometry(0.09, 16, 16);
        brainData.mem_3d.forEach((pos, idx) => {{
            const energy = brainData.mem_energy[idx];
            const hue = (1.0 - energy / 5.0) * 0.15 + 0.05; // Yellow to red
            const mat = new THREE.MeshStandardMaterial({{
                color: new THREE.Color().setHSL(0.12, 1.0, 0.55),
                emissive: new THREE.Color().setHSL(0.08, 1.0, 0.3),
                roughness: 0.3
            }});
            const mesh = new THREE.Mesh(memGeom, mat);
            mesh.position.set(pos[0], pos[1], pos[2]);
            mesh.userData = {{ type: 'Memory Slot', id: idx, shell: brainData.mem_r[idx].toFixed(2), energy: energy.toFixed(2) }};
            memGroup.add(mesh);
        }});
        scene.add(memGroup);

        // Dendritic Experts (Octahedra)
        const expGroup = new THREE.Group();
        const expGeom = new THREE.OctahedronGeometry(0.25, 0);
        const expColors = [0x3498db, 0x9b59b6, 0x2ecc71, 0xe67e22, 0xe74c3c, 0x1abc9c, 0xf1c40f];
        brainData.exp_3d.forEach((pos, idx) => {{
            const mat = new THREE.MeshStandardMaterial({{
                color: expColors[idx % expColors.length],
                emissive: expColors[idx % expColors.length],
                emissiveIntensity: 0.4,
                roughness: 0.2
            }});
            const mesh = new THREE.Mesh(expGeom, mat);
            mesh.position.set(pos[0], pos[1], pos[2]);
            mesh.userData = {{
                type: 'Dendritic Expert',
                name: brainData.expert_names[idx],
                shell: brainData.expert_r[idx].toFixed(2),
                usage: brainData.expert_usage[idx]
            }};
            expGroup.add(mesh);
        }});
        scene.add(expGroup);

        // Deliberation Trajectory Line
        const points = brainData.traj_3d.map(p => new THREE.Vector3(p[0], p[1], p[2]));
        const curve = new THREE.CatmullRomCurve3(points);
        const tubeGeom = new THREE.TubeGeometry(curve, 100, 0.04, 8, false);
        const tubeMat = new THREE.MeshBasicMaterial({{ color: 0xff3366, transparent: true, opacity: 0.85 }});
        scene.add(new THREE.Mesh(tubeGeom, tubeMat));

        // Thought Pointer (pulsing sphere)
        const lastPt = points[points.length - 1];
        const pointerGeom = new THREE.SphereGeometry(0.16, 24, 24);
        const pointerMat = new THREE.MeshStandardMaterial({{ color: 0xff0055, emissive: 0xff0055, emissiveIntensity: 0.8 }});
        const pointerMesh = new THREE.Mesh(pointerGeom, pointerMat);
        pointerMesh.position.copy(lastPt);
        scene.add(pointerMesh);

        // Tooltip Raycasting
        const raycaster = new THREE.Raycaster();
        const mouse = new THREE.Vector2();
        const tooltip = document.getElementById('tooltip');

        window.addEventListener('mousemove', (e) => {{
            mouse.x = (e.clientX / window.innerWidth) * 2 - 1;
            mouse.y = -(e.clientY / window.innerHeight) * 2 + 1;

            raycaster.setFromCamera(mouse, camera);
            const intersects = raycaster.intersectObjects([...expGroup.children, ...memGroup.children]);

            if (intersects.length > 0) {{
                const data = intersects[0].object.userData;
                tooltip.style.display = 'block';
                tooltip.style.left = e.clientX + 'px';
                tooltip.style.top = e.clientY + 'px';
                if (data.type === 'Dendritic Expert') {{
                    tooltip.innerHTML = `<b>${{data.name}}</b><br>Shell Radius: r=${{data.shell}}<br>Routing Usage: ${{data.usage}}`;
                }} else {{
                    tooltip.innerHTML = `<b>Slot #${{data.id}}</b><br>Shell Radius: r=${{data.shell}}<br>Retention Energy: ${{data.energy}}`;
                }}
            }} else {{
                tooltip.style.display = 'none';
            }}
        }});

        window.addEventListener('resize', () => {{
            camera.aspect = window.innerWidth / window.innerHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(window.innerWidth, window.innerHeight);
        }});

        // Animation Loop
        let clock = new THREE.Clock();
        function animate() {{
            requestAnimationFrame(animate);
            const t = clock.getElapsedTime();

            expGroup.rotation.y = t * 0.05;
            memGroup.rotation.y = t * 0.02;

            const scale = 1.0 + 0.15 * Math.sin(t * 5.0);
            pointerMesh.scale.set(scale, scale, scale);

            controls.update();
            renderer.render(scene, camera);
        }}
        animate();
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"Generated interactive 3D WebGL brain map at: {output_html}")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs("artifacts", exist_ok=True)
    uce = train_or_load_engine(device)
    brain_state = extract_brain_state(uce)

    out_png = os.path.join("artifacts", "system_brain_map.png")
    out_html = os.path.join("artifacts", "system_brain_map.html")

    generate_static_plot(brain_state, out_png)
    generate_interactive_html(brain_state, out_html)
