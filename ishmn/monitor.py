"""
Monitor and Dashboard Generator for Self-Evolving Hyperspace System.

Generates:
1. Matplotlib multi-panel metric summary: artifacts/evolution_metrics.png
2. Interactive 3D WebGL lifelong dashboard: artifacts/evolution_dashboard.html
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def render_evolution_plots(telemetry_file: str, output_image: str):
    with open(telemetry_file, "r") as f:
        data = json.load(f)

    history = data["history"]
    cycles = history["cycle"]

    fig, axs = plt.subplots(2, 2, figsize=(14, 9), dpi=150)
    fig.suptitle("Lifelong Self-Evolving Loop: Telemetry & Memory Dynamics", fontsize=14, fontweight='bold')

    # 1. Surprise / Curiosity & Prediction Error
    axs[0, 0].plot(cycles, history["surprise"], color='#f97316', label='Riemannian Surprise (Curiosity)', linewidth=2)
    axs[0, 0].plot(cycles, history["prediction_error"], color='#3b82f6', label='World Model Pred Error', linestyle='--', linewidth=1.8)
    axs[0, 0].set_title("Intrinsic Curiosity & Predictive Foresight", fontsize=11, fontweight='bold')
    axs[0, 0].set_xlabel("Evolution Cycle")
    axs[0, 0].set_ylabel("Error / Surprise")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    # 2. Infinite Shell Growth (r_max and r_min)
    axs[0, 1].plot(cycles, history["max_shell_radius"], color='#a855f7', label='Outer Shell Horizon (r_max)', linewidth=2)
    axs[0, 1].plot(cycles, history["min_shell_radius"], color='#06b6d4', label='Core Persistent Shell (r_min)', linewidth=1.8)
    axs[0, 1].set_title("Concentric Hyperspace Shell Horizon (r -> inf)", fontsize=11, fontweight='bold')
    axs[0, 1].set_xlabel("Evolution Cycle")
    axs[0, 1].set_ylabel("Radial Distance r")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    # 3. Active Memory Slots
    axs[1, 0].step(cycles, history["active_slots"], color='#22c55e', where='mid', linewidth=2)
    axs[1, 0].set_title("Active Memory Slots (Dynamic Allocation & Compaction)", fontsize=11, fontweight='bold')
    axs[1, 0].set_xlabel("Evolution Cycle")
    axs[1, 0].set_ylabel("Slot Count (N)")
    axs[1, 0].grid(True, alpha=0.3)

    # 4. Average Retention Energy
    axs[1, 1].plot(cycles, history["avg_retention_energy"], color='#eab308', linewidth=2)
    axs[1, 1].set_title("Consolidated Memory Retention Energy", fontsize=11, fontweight='bold')
    axs[1, 1].set_xlabel("Evolution Cycle")
    axs[1, 1].set_ylabel("Synaptic Energy")
    axs[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"[Monitor] Evolution metrics plot saved to: {output_image}")

def generate_evolution_dashboard_html(telemetry_file: str, output_html: str):
    with open(telemetry_file, "r") as f:
        data = json.load(f)

    json_str = json.dumps(data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Autonomous Self-Evolving Hyperspace Memory Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: #05070e;
            color: #d1d9e6;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            display: flex;
        }}
        #viewport {{ flex: 1; height: 100%; position: relative; }}
        #canvas-container {{ width: 100%; height: 100%; }}
        #sidebar {{
            width: 420px;
            background: rgba(10, 14, 26, 0.95);
            backdrop-filter: blur(14px);
            border-left: 1px solid rgba(70, 120, 240, 0.25);
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
            color: #58a6ff;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .badge {{
            font-size: 0.68rem;
            background: #238636;
            color: #fff;
            padding: 2px 8px;
            border-radius: 10px;
        }}
        .badge.dream {{ background: #8957e5; }}
        .panel {{
            background: rgba(18, 25, 46, 0.7);
            border: 1px solid rgba(70, 120, 240, 0.2);
            border-radius: 8px;
            padding: 14px;
        }}
        .panel-title {{
            font-size: 0.75rem;
            font-weight: 600;
            color: #8b949e;
            text-transform: uppercase;
            margin-bottom: 8px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 8px;
        }}
        .metric-card {{
            background: rgba(8, 12, 22, 0.7);
            border-radius: 6px;
            padding: 10px;
            border-left: 3px solid #58a6ff;
        }}
        .metric-val {{
            font-size: 1.15rem;
            font-weight: 700;
            color: #ffffff;
            font-family: monospace;
        }}
        .metric-lbl {{ font-size: 0.7rem; color: #8b949e; margin-top: 2px; }}
        .controls {{ display: flex; gap: 8px; align-items: center; }}
        button {{
            background: #238636;
            color: white;
            border: none;
            padding: 7px 14px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.8rem;
            transition: 0.2s;
        }}
        button:hover {{ background: #2ea043; }}
        input[type="range"] {{ flex: 1; accent-color: #58a6ff; cursor: pointer; }}
        .hud-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
            background: rgba(8, 12, 24, 0.8);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(80, 130, 240, 0.3);
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 0.85rem;
            color: #e2e8f0;
            line-height: 1.5;
        }}
        .hud-title {{ font-weight: 700; color: #60a5fa; margin-bottom: 4px; }}
        #energy-bars {{
            display: flex;
            gap: 3px;
            height: 44px;
            align-items: flex-end;
            margin-top: 6px;
            background: rgba(0,0,0,0.3);
            padding: 4px;
            border-radius: 4px;
        }}
        .energy-bar {{
            flex: 1;
            background: #eab308;
            border-radius: 2px 2px 0 0;
            min-height: 2px;
            transition: height 0.15s ease;
        }}
    </style>
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="viewport">
        <div id="canvas-container"></div>
        <div class="hud-overlay">
            <div class="hud-title">Self-Evolving Infinite Spherical Hyperspace</div>
            <div>Phase: <b id="hud-phase" style="color: #4ade80;">Wake</b></div>
            <div>Outer Shell Horizon: <b id="hud-shell">r = 4.00</b></div>
            <div>Curiosity Surprise: <b id="hud-surprise">0.00</b></div>
            <div style="font-size: 0.72rem; color: #94a3b8; margin-top: 4px;">&bull; Left Drag: Orbit | Right Drag: Pan | Scroll: Zoom</div>
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>Lifelong Evolution <span id="cycle-badge" class="badge">ACTIVE</span></h1>
            <p style="font-size: 0.76rem; color: #8b949e; margin-top: 3px;">
                Universal Multi-Network Perception, Dreaming & Infinite Shell Expansion
            </p>
        </div>

        <div class="panel">
            <div class="panel-title">Cycle Scrubbing</div>
            <div class="controls" style="margin-bottom: 10px;">
                <button id="btn-play">Pause</button>
                <span id="cycle-display" style="font-family: monospace; font-size: 0.85rem;">Cycle 1/1</span>
            </div>
            <input type="range" id="cycle-slider" min="0" max="0" value="0">
        </div>

        <div class="panel">
            <div class="panel-title">Evolutionary State Telemetry</div>
            <div class="metric-grid">
                <div class="metric-card" style="border-left-color: #f97316;">
                    <div class="metric-val" id="val-surprise">0.00</div>
                    <div class="metric-lbl">Surprise (Curiosity)</div>
                </div>
                <div class="metric-card" style="border-left-color: #3b82f6;">
                    <div class="metric-val" id="val-pred">0.00</div>
                    <div class="metric-lbl">World Model Loss</div>
                </div>
                <div class="metric-card" style="border-left-color: #a855f7;">
                    <div class="metric-val" id="val-max-r">0.00</div>
                    <div class="metric-lbl">Outer Shell (r<sub>max</sub>)</div>
                </div>
                <div class="metric-card" style="border-left-color: #22c55e;">
                    <div class="metric-val" id="val-slots">0</div>
                    <div class="metric-lbl">Active Memory Slots</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">Synaptic Retention Energy (Memory Consolidation)</div>
            <div id="energy-bars"></div>
        </div>

        <div class="panel" style="font-size: 0.78rem; line-height: 1.5; color: #94a3b8;">
            <div class="panel-title" style="color: #cbd5e1;">How Self-Evolution Works:</div>
            <div>&bull; <b>Perception</b>: Any text, sensor waveform, or vector is mapped into S<sup>D-1</sup> &times; &reals;<sup>+</sup>.</div>
            <div>&bull; <b>Predictive Model</b>: Anticipates trajectory; discrepancy triggers Riemannian surprise.</div>
            <div>&bull; <b>Shell Spawning</b>: High surprise expands memory outwards into infinite concentric shells (r &rarr; &infin;).</div>
            <div>&bull; <b>Dreaming Replay</b>: Periodically consolidates high-energy memories inward towards core shells and compacts duplicate clusters.</div>
        </div>
    </div>

    <script>
        const simData = {json_str};
        const history = simData.history;
        const traces = simData.recent_traces;
        const totalCycles = history.cycle.length;

        let currentIdx = 0;
        let isPlaying = true;
        let lastFrameTime = 0;

        const slider = document.getElementById('cycle-slider');
        slider.max = totalCycles - 1;
        const cycleDisplay = document.getElementById('cycle-display');
        const btnPlay = document.getElementById('btn-play');
        const hudPhase = document.getElementById('hud-phase');
        const hudShell = document.getElementById('hud-shell');
        const hudSurprise = document.getElementById('hud-surprise');
        const valSurprise = document.getElementById('val-surprise');
        const valPred = document.getElementById('val-pred');
        const valMaxR = document.getElementById('val-max-r');
        const valSlots = document.getElementById('val-slots');
        const energyBarsContainer = document.getElementById('energy-bars');

        // Setup Three.js
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x05070e, 0.02);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(14, 10, 16);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        scene.add(new THREE.AmbientLight(0xffffff, 0.8));
        const dirLight = new THREE.DirectionalLight(0x93c5fd, 1.3);
        dirLight.position.set(15, 25, 15);
        scene.add(dirLight);

        // Concentric Spheres
        const shellMeshes = [];
        function rebuildShells(maxRadius) {{
            shellMeshes.forEach(m => scene.remove(m));
            shellMeshes.length = 0;

            const shellCount = 4;
            for (let i = 1; i <= shellCount; i++) {{
                const r = (maxRadius / shellCount) * i;
                const geom = new THREE.SphereGeometry(r, 26, 18);
                const mat = new THREE.MeshBasicMaterial({{
                    color: [0x388bfd, 0x10b981, 0xf59e0b, 0xa855f7][(i - 1) % 4],
                    wireframe: true,
                    transparent: true,
                    opacity: 0.13 - i * 0.02
                }});
                const mesh = new THREE.Mesh(geom, mat);
                scene.add(mesh);
                shellMeshes.push(mesh);
            }}
        }}

        // Memory slot nodes
        const slotGroup = new THREE.Group();
        scene.add(slotGroup);

        function project3D(u, r) {{
            let x = u[0] || 0, y = u[1] || 0, z = u[2] || 0;
            const norm = Math.hypot(x, y, z) || 1.0;
            return new THREE.Vector3((x / norm) * r, (y / norm) * r, (z / norm) * r);
        }}

        function updateCycle(idx) {{
            currentIdx = Math.max(0, Math.min(totalCycles - 1, idx));
            slider.value = currentIdx;
            const cycleNum = history.cycle[currentIdx];
            cycleDisplay.innerText = `Cycle ${{cycleNum}}/${{totalCycles}}`;

            const phase = history.phase[currentIdx];
            hudPhase.innerText = phase;
            hudPhase.style.color = phase.includes("Dream") ? "#c084fc" : "#4ade80";

            valSurprise.innerText = history.surprise[currentIdx].toFixed(3);
            valPred.innerText = history.prediction_error[currentIdx].toFixed(3);
            valMaxR.innerText = history.max_shell_radius[currentIdx].toFixed(2);
            valSlots.innerText = history.active_slots[currentIdx];

            hudShell.innerText = `r = ${{history.max_shell_radius[currentIdx].toFixed(2)}}`;
            hudSurprise.innerText = history.surprise[currentIdx].toFixed(3);

            rebuildShells(history.max_shell_radius[currentIdx]);

            // Match closest trace
            const traceIdx = Math.min(currentIdx, traces.length - 1);
            if (traces[traceIdx]) {{
                const tr = traces[traceIdx];
                // Update 3D slot meshes
                while (slotGroup.children.length < tr.mem_u.length) {{
                    const geom = new THREE.SphereGeometry(0.2, 12, 12);
                    const mat = new THREE.MeshStandardMaterial({{ color: 0xfacc15, emissive: 0xca8a04, emissiveIntensity: 0.4 }});
                    slotGroup.add(new THREE.Mesh(geom, mat));
                }}
                while (slotGroup.children.length > tr.mem_u.length) {{
                    slotGroup.remove(slotGroup.children[slotGroup.children.length - 1]);
                }}

                energyBarsContainer.innerHTML = '';
                for (let i = 0; i < tr.mem_u.length; i++) {{
                    const pos = project3D(tr.mem_u[i], tr.mem_r[i]);
                    const mesh = slotGroup.children[i];
                    mesh.position.copy(pos);
                    const eng = (tr.mem_energy && tr.mem_energy[i]) ? tr.mem_energy[i] : 1.0;
                    mesh.scale.setScalar(0.7 + eng * 0.4);

                    const bar = document.createElement('div');
                    bar.className = 'energy-bar';
                    bar.style.height = `${{Math.min(100, eng * 25)}}%`;
                    energyBarsContainer.appendChild(bar);
                }}
            }}
        }}

        slider.addEventListener('input', (e) => updateCycle(parseInt(e.target.value)));
        btnPlay.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            btnPlay.innerText = isPlaying ? "Pause" : "Play";
        }});

        function animate(timestamp) {{
            requestAnimationFrame(animate);
            if (isPlaying && timestamp - lastFrameTime > 320) {{
                lastFrameTime = timestamp;
                let nextIdx = currentIdx + 1;
                if (nextIdx >= totalCycles) nextIdx = 0;
                updateCycle(nextIdx);
            }}
            scene.rotation.y += 0.0012;
            controls.update();
            renderer.render(scene, camera);
        }}

        updateCycle(0);
        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[Monitor] Interactive self-evolution dashboard generated at: {output_html}")
    print("[Monitor] Unified 3D Hyperspace Studio V3 available at: artifacts/dashboard_v3.html")

if __name__ == "__main__":
    telemetry_file = "artifacts/evolution_telemetry.json"
    if os.path.exists(telemetry_file):
        render_evolution_plots(telemetry_file, "artifacts/evolution_metrics.png")
        generate_evolution_dashboard_html(telemetry_file, "artifacts/evolution_dashboard.html")
    else:
        print(f"Telemetry file {telemetry_file} not found. Run self_evolving_loop.py first.")
    print("[*] 3D Hyperspace Studio V3 (Inverted Mobius, Spherical Memory, Multi-Agent Bus) ready at: artifacts/dashboard_v3.html")
