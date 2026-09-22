"""
Demonstration of Continuous Self-Loop with Dynamic MoE and Triggered Emissions.

Simulates a living continuous dynamical loop:
- Ongoing endogenous hyperspherical oscillations
- Sporadic external perturbations (inputs injected asynchronously)
- Autonomous Neurogenesis (spawning dendritic experts mid-run upon novelty)
- Triggered Emissions (outputs only emerge when consensus / attractor forms)
- Idle Sleep Consolidation (merging & pruning)
- Telemetry export and 3D WebGL Dashboard generation
"""

import json
import os
import random
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ishmn.continuous_self_loop import ContinuousSelfLoop, TriggeredEmission

def run_continuous_loop_demo(
    total_ticks: int = 80,
    save_dir: str = "artifacts"
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Continuous Loop Demo] Initializing on {device}...")
    os.makedirs(save_dir, exist_ok=True)

    loop = ContinuousSelfLoop(
        d_model=32,
        d_hidden=64,
        hyperspace_dim=8,
        num_memory_slots=16,
        initial_experts=2,
        max_experts=12,
        emission_threshold=0.52,
        device=device
    )

    # Define scheduled external perturbations (diverse domains)
    perturbation_schedule = {
        5: {"tag": "Quantum Harmonic Wave", "modality": "continuous", "data": torch.randn(1, 16)},
        18: {"tag": "Linguistic Grammar Concept", "modality": "tokens", "data": torch.randint(0, 500, (1, 8))},
        35: {"tag": "Biomedical Gene Expression", "modality": "continuous", "data": torch.randn(1, 28)},
        50: {"tag": "Astrophysical Metric", "modality": "continuous", "data": torch.randn(1, 20)},
        65: {"tag": "Symbolic Logic Stream", "modality": "tokens", "data": torch.randint(0, 500, (1, 6))}
    }

    emissions_log = []
    print(f"\n--- Running Living Self-Loop for {total_ticks} ticks ---")

    for tick in range(1, total_ticks + 1):
        # 1. Check if external perturbation is injected at this tick
        if tick in perturbation_schedule:
            event = perturbation_schedule[tick]
            print(f"\n[Tick {tick:2d}] >>> INJECTING EXTERNAL PERTURBATION: {event['tag']} ({event['modality']})")
            loop.inject_input(event["data"], modality=event["modality"], tag=event["tag"])

        # 2. Step the continuous loop
        emission = loop.loop_tick()

        # 3. Check if an output was triggered from the loop
        if emission is not None:
            print(
                f"  >>> [TRIGGERED EMISSION Tick {tick:2d}] {emission.trigger_reason} | "
                f"Active Experts: {emission.active_experts} | Radius r: {emission.coordinate_r:.2f}"
            )
            emissions_log.append({
                "step": emission.step,
                "confidence": emission.confidence,
                "reason": emission.trigger_reason,
                "active_experts": emission.active_experts,
                "coord_r": emission.coordinate_r,
                "coord_u": emission.coordinate_u
            })
        else:
            # Silent internal deliberation / endogenous oscillation
            if tick % 10 == 0:
                print(f"  [Tick {tick:2d}] Loop running internally | Mode: {loop.telemetry['mode'][-1]} | Experts: {loop.moe.num_experts}")

    print(f"\n[Continuous Loop Demo] Completed {total_ticks} ticks!")
    print(f"Total Triggered Emissions: {len(emissions_log)} | Final Active Experts: {loop.moe.num_experts}")

    # Export telemetry
    telemetry_data = {
        "telemetry": loop.telemetry,
        "emissions": emissions_log,
        "final_state": {
            "num_experts": loop.moe.num_experts,
            "expert_anchors_u": loop.moe.expert_u.cpu().numpy().tolist(),
            "expert_anchors_r": loop.moe.expert_r.squeeze(-1).cpu().numpy().tolist(),
            "expert_usage": loop.moe.expert_usage.cpu().numpy().tolist(),
            "mem_slots_count": loop.mem_state["u"].size(1),
            "mem_u": loop.mem_state["u"][0].cpu().numpy().tolist(),
            "mem_r": loop.mem_state["r"][0, :, 0].cpu().numpy().tolist()
        }
    }

    telemetry_file = os.path.join(save_dir, "continuous_loop_telemetry.json")
    with open(telemetry_file, "w") as f:
        json.dump(telemetry_data, f, indent=2)

    # Generate Matplotlib summary plot
    render_continuous_loop_plot(telemetry_data, os.path.join(save_dir, "continuous_loop_metrics.png"))

    # Generate Interactive 3D WebGL Dashboard V2
    generate_dashboard_v2_html(telemetry_data, os.path.join(save_dir, "dashboard_v2.html"))

    return telemetry_data

def render_continuous_loop_plot(data, output_path):
    tel = data["telemetry"]
    steps = tel["step"]
    emissions = data["emissions"]
    emission_steps = [e["step"] for e in emissions]
    emission_confs = [e["confidence"] for e in emissions]

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.suptitle("Continuous Autonomous Self-Loop: Dynamic MoE & Triggered Emissions", fontsize=14, fontweight='bold')

    # 1. Trigger Confidence & Output Emissions
    axs[0, 0].plot(steps, tel["trigger_confidence"], color='#38bdf8', label='Internal Readiness / Confidence', linewidth=1.8)
    axs[0, 0].scatter(emission_steps, emission_confs, color='#f43f5e', s=80, zorder=5, label='Triggered Emission Fired')
    axs[0, 0].axhline(0.68, color='#f43f5e', linestyle='--', alpha=0.6, label='Emission Threshold (0.68)')
    axs[0, 0].set_title("Triggered Emission Dynamics (Consensus Threshold)", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Loop Tick")
    axs[0, 0].set_ylabel("Trigger Confidence")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend(loc='lower right')

    # 2. Dynamic MoE Neurogenesis (Active Experts count)
    axs[0, 1].step(steps, tel["active_experts"], color='#a855f7', where='mid', linewidth=2.2)
    axs[0, 1].set_title("Autonomous Expert Neurogenesis (Clonal Mitosis)", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("Loop Tick")
    axs[0, 1].set_ylabel("Active Dendritic Experts (E)")
    axs[0, 1].grid(True, alpha=0.3)

    # 3. Hyperspace Trajectory Radius
    axs[1, 0].plot(steps, tel["r_trajectory"], color='#eab308', label='Loop Trajectory Radius (r_t)', linewidth=2)
    axs[1, 0].plot(steps, tel["max_shell"], color='#22c55e', linestyle=':', label='Outer Memory Shell (r_max)', linewidth=1.8)
    axs[1, 0].set_title("Hyperspace Radial Trajectory & Shell Horizon", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("Loop Tick")
    axs[1, 0].set_ylabel("Radial Coordinate r")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    # 4. Expert Usage Distribution
    expert_usage = data["final_state"]["expert_usage"]
    expert_indices = list(range(len(expert_usage)))
    axs[1, 1].bar(expert_indices, expert_usage, color='#6366f1', alpha=0.85, edgecolor='black')
    axs[1, 1].set_title("Final Dendritic Expert Workload Allocation", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("Expert Index")
    axs[1, 1].set_ylabel("Routing Usage Count")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[Continuous Loop Demo] Metrics plot saved to: {output_path}")

def generate_dashboard_v2_html(data, output_html):
    data_json = json.dumps(data)
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dynamic MoE & Continuous Self-Loop Visualizer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: #050811;
            color: #e2e8f0;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            display: flex;
        }}
        #viewport {{ flex: 1; height: 100%; position: relative; }}
        #canvas-container {{ width: 100%; height: 100%; }}
        #sidebar {{
            width: 440px;
            background: rgba(11, 17, 33, 0.96);
            backdrop-filter: blur(14px);
            border-left: 1px solid rgba(80, 130, 240, 0.25);
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 16px;
            box-shadow: -10px 0 30px rgba(0,0,0,0.6);
            z-index: 10;
        }}
        h1 {{
            font-size: 1.15rem;
            color: #38bdf8;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .badge {{
            font-size: 0.68rem;
            background: #8b5cf6;
            color: #fff;
            padding: 2px 8px;
            border-radius: 10px;
        }}
        .panel {{
            background: rgba(18, 26, 48, 0.7);
            border: 1px solid rgba(80, 130, 240, 0.2);
            border-radius: 8px;
            padding: 14px;
        }}
        .panel-title {{
            font-size: 0.75rem;
            font-weight: 600;
            color: #94a3b8;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .metric-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 8px; }}
        .metric-card {{
            background: rgba(10, 15, 28, 0.7);
            border-radius: 6px;
            padding: 10px;
            border-left: 3px solid #38bdf8;
        }}
        .metric-val {{ font-size: 1.15rem; font-weight: 700; color: #ffffff; font-family: monospace; }}
        .metric-lbl {{ font-size: 0.7rem; color: #94a3b8; margin-top: 2px; }}
        .controls {{ display: flex; gap: 8px; align-items: center; }}
        button {{
            background: #2563eb;
            color: white;
            border: none;
            padding: 7px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.8rem;
            transition: 0.2s;
        }}
        button:hover {{ background: #1d4ed8; }}
        input[type="range"] {{ flex: 1; accent-color: #38bdf8; cursor: pointer; }}
        .hud-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
            background: rgba(10, 16, 30, 0.85);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(80, 130, 240, 0.3);
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 0.85rem;
            color: #e2e8f0;
            line-height: 1.5;
        }}
        .hud-title {{ font-weight: 700; color: #38bdf8; margin-bottom: 4px; }}
        .emission-feed {{
            display: flex;
            flex-direction: column;
            gap: 6px;
            max-height: 140px;
            overflow-y: auto;
        }}
        .emission-item {{
            background: rgba(15, 23, 42, 0.8);
            border-left: 3px solid #f43f5e;
            padding: 6px 10px;
            border-radius: 4px;
            font-size: 0.75rem;
            line-height: 1.4;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
            margin-bottom: 4px;
        }}
        .dot {{ width: 10px; height: 10px; border-radius: 50%; display: inline-block; }}
    </style>
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="viewport">
        <div id="canvas-container"></div>
        <div class="hud-overlay">
            <div class="hud-title">Living Continuous Self-Loop in Infinite Hyperspace</div>
            <div>Mode: <b id="hud-mode" style="color: #38bdf8;">Endogenous Oscillation</b></div>
            <div>Readiness / Confidence: <b id="hud-conf">0.00</b></div>
            <div>Dendritic Experts (MoE): <b id="hud-experts">2</b></div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">&bull; Left Drag: Orbit | Right Drag: Pan | Scroll: Zoom</div>
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>Continuous Self-Loop <span class="badge">DYNAMIC MOE</span></h1>
            <p style="font-size: 0.76rem; color: #94a3b8; margin-top: 3px;">
                Neurogenesis, Clonal Mitosis & Triggered Emergent Emissions
            </p>
        </div>

        <div class="panel">
            <div class="panel-title">Loop Clock Tick</div>
            <div class="controls" style="margin-bottom: 10px;">
                <button id="btn-play">Pause</button>
                <span id="tick-display" style="font-family: monospace; font-size: 0.85rem;">Tick 1/1</span>
            </div>
            <input type="range" id="tick-slider" min="0" max="0" value="0">
        </div>

        <div class="panel">
            <div class="panel-title">System Metrics</div>
            <div class="metric-grid">
                <div class="metric-card" style="border-left-color: #38bdf8;">
                    <div class="metric-val" id="val-conf">0.00</div>
                    <div class="metric-lbl">Trigger Confidence</div>
                </div>
                <div class="metric-card" style="border-left-color: #a855f7;">
                    <div class="metric-val" id="val-experts">2</div>
                    <div class="metric-lbl">Active Experts (E)</div>
                </div>
                <div class="metric-card" style="border-left-color: #eab308;">
                    <div class="metric-val" id="val-radius">1.50</div>
                    <div class="metric-lbl">Trajectory Radius (r<sub>t</sub>)</div>
                </div>
                <div class="metric-card" style="border-left-color: #f43f5e;">
                    <div class="metric-val" id="val-emissions">0</div>
                    <div class="metric-lbl">Triggered Emissions</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Triggered Output Emissions Feed</div>
            <div class="emission-feed" id="emission-feed"></div>
        </div>

        <div class="panel">
            <div class="panel-title">3D Hypersphere Map Legend</div>
            <div class="legend-item">
                <span class="dot" style="background: #38bdf8; box-shadow: 0 0 6px #38bdf8;"></span>
                <span><b>Current Loop Trajectory Pointer</b></span>
            </div>
            <div class="legend-item">
                <span class="dot" style="background: #a855f7; box-shadow: 0 0 6px #a855f7;"></span>
                <span><b>Dendritic Expert Anchors</b> (Dynamic MoE)</span>
            </div>
            <div class="legend-item">
                <span class="dot" style="background: #eab308;"></span>
                <span><b>Memory Slots</b> (Concentric Shells)</span>
            </div>
            <div class="legend-item">
                <span class="dot" style="background: #f43f5e; box-shadow: 0 0 8px #f43f5e;"></span>
                <span><b>Output Emission Trigger Point</b></span>
            </div>
        </div>
    </div>

    <script>
        const simData = {data_json};
        const tel = simData.telemetry;
        const emissions = simData.emissions;
        const finalState = simData.final_state;
        const totalTicks = tel.step.length;

        let currentIdx = 0;
        let isPlaying = true;
        let lastFrameTime = 0;

        const slider = document.getElementById('tick-slider');
        slider.max = totalTicks - 1;
        const tickDisplay = document.getElementById('tick-display');
        const btnPlay = document.getElementById('btn-play');
        const hudMode = document.getElementById('hud-mode');
        const hudConf = document.getElementById('hud-conf');
        const hudExperts = document.getElementById('hud-experts');
        const valConf = document.getElementById('val-conf');
        const valExperts = document.getElementById('val-experts');
        const valRadius = document.getElementById('val-radius');
        const valEmissions = document.getElementById('val-emissions');
        const emissionFeed = document.getElementById('emission-feed');

        // Populate emissions feed
        emissions.forEach(e => {{
            const div = document.createElement('div');
            div.className = 'emission-item';
            div.innerHTML = `<b>Tick ${{e.step}}</b>: ${{e.reason}} <br><span style="color: #94a3b8;">Radius: ${{e.coord_r.toFixed(2)}} | Conf: ${{e.confidence.toFixed(3)}}</span>`;
            emissionFeed.appendChild(div);
        }});

        // Three.js Scene Setup
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x050811, 0.02);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(14, 11, 15);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        scene.add(new THREE.AmbientLight(0xffffff, 0.85));
        const dirLight = new THREE.DirectionalLight(0x93c5fd, 1.4);
        dirLight.position.set(15, 25, 15);
        scene.add(dirLight);

        // Concentric Spheres
        const shells = [2.0, 3.5, 5.0, 6.5];
        shells.forEach((r, idx) => {{
            const geom = new THREE.SphereGeometry(r, 26, 18);
            const mat = new THREE.MeshBasicMaterial({{
                color: [0x38bdf8, 0x10b981, 0xf59e0b, 0xa855f7][idx % 4],
                wireframe: true,
                transparent: true,
                opacity: 0.12 - idx * 0.015
            }});
            scene.add(new THREE.Mesh(geom, mat));
        }});

        function project3D(u, r) {{
            let x = u[0] || 0, y = u[1] || 0, z = u[2] || 0;
            const norm = Math.hypot(x, y, z) || 1.0;
            return new THREE.Vector3((x / norm) * r, (y / norm) * r, (z / norm) * r);
        }}

        // Loop Trajectory Pointer
        const pointerGeom = new THREE.SphereGeometry(0.35, 16, 16);
        const pointerMat = new THREE.MeshStandardMaterial({{
            color: 0x38bdf8,
            emissive: 0x0284c7,
            emissiveIntensity: 0.9
        }});
        const pointerMesh = new THREE.Mesh(pointerGeom, pointerMat);
        scene.add(pointerMesh);

        // Expert Anchors (Dynamic MoE)
        const expertGroup = new THREE.Group();
        scene.add(expertGroup);
        finalState.expert_anchors_u.forEach((u, i) => {{
            const r = finalState.expert_anchors_r[i];
            const pos = project3D(u, r);
            const geom = new THREE.OctahedronGeometry(0.35);
            const mat = new THREE.MeshStandardMaterial({{
                color: 0xa855f7,
                emissive: 0x7e22ce,
                emissiveIntensity: 0.6
            }});
            const mesh = new THREE.Mesh(geom, mat);
            mesh.position.copy(pos);
            expertGroup.add(mesh);
        }});

        // Memory Slots
        const memoryGroup = new THREE.Group();
        scene.add(memoryGroup);
        finalState.mem_u.forEach((u, i) => {{
            const r = finalState.mem_r[i];
            const pos = project3D(u, r);
            const geom = new THREE.SphereGeometry(0.18, 12, 12);
            const mat = new THREE.MeshStandardMaterial({{ color: 0xeab308, emissive: 0xca8a04, emissiveIntensity: 0.3 }});
            const mesh = new THREE.Mesh(geom, mat);
            mesh.position.copy(pos);
            memoryGroup.add(mesh);
        }});

        function updateTick(idx) {{
            currentIdx = Math.max(0, Math.min(totalTicks - 1, idx));
            slider.value = currentIdx;
            const tickNum = tel.step[currentIdx];
            tickDisplay.innerText = `Tick ${{tickNum}}/${{totalTicks}}`;

            const mode = tel.mode[currentIdx];
            const conf = tel.trigger_confidence[currentIdx];
            const experts = tel.active_experts[currentIdx];
            const r = tel.r_trajectory[currentIdx];

            hudMode.innerText = mode;
            hudMode.style.color = mode.includes("Perturbed") ? "#f43f5e" : (mode.includes("Sleep") ? "#c084fc" : "#38bdf8");
            hudConf.innerText = conf.toFixed(3);
            hudExperts.innerText = experts;

            valConf.innerText = conf.toFixed(3);
            valExperts.innerText = experts;
            valRadius.innerText = r.toFixed(2);

            // Count emissions up to this tick
            const emissionsSoFar = emissions.filter(e => e.step <= tickNum).length;
            valEmissions.innerText = emissionsSoFar;

            // Update Trajectory Pointer
            const u = tel.u_trajectory[currentIdx];
            const pos = project3D(u, r);
            pointerMesh.position.copy(pos);

            // Pulse pointer if emission triggered
            if (conf >= 0.68) {{
                pointerMesh.material.color.setHex(0xf43f5e);
                pointerMesh.material.emissive.setHex(0xe11d48);
                pointerMesh.scale.setScalar(1.5);
            }} else {{
                pointerMesh.material.color.setHex(0x38bdf8);
                pointerMesh.material.emissive.setHex(0x0284c7);
                pointerMesh.scale.setScalar(1.0);
            }}

            // Show active experts up to current count
            expertGroup.children.forEach((mesh, i) => {{
                mesh.visible = (i < experts);
            }});
        }}

        slider.addEventListener('input', (e) => updateTick(parseInt(e.target.value)));
        btnPlay.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            btnPlay.innerText = isPlaying ? "Pause" : "Play";
        }});

        function animate(timestamp) {{
            requestAnimationFrame(animate);
            if (isPlaying && timestamp - lastFrameTime > 260) {{
                lastFrameTime = timestamp;
                let nextIdx = currentIdx + 1;
                if (nextIdx >= totalTicks) nextIdx = 0;
                updateTick(nextIdx);
            }}
            scene.rotation.y += 0.0012;
            controls.update();
            renderer.render(scene, camera);
        }}

        updateTick(0);
        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[Continuous Loop Demo] Interactive 3D WebGL Dashboard V2 generated at: {output_html}")

if __name__ == "__main__":
    run_continuous_loop_demo(total_ticks=80, save_dir="artifacts")
