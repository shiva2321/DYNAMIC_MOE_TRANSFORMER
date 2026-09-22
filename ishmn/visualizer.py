"""
Visualization of Memory Navigation in Infinite Spherical Hyperspace.

Features:
- Projects D-dimensional spherical coordinates (u, r) onto 3D via PCA/SVD
- Renders concentric infinite spherical shells (r = 1, 2, 3...)
- Plots trajectory of Read Head, Write Head, and steered memory slots
- Saves high-res 3D static plot as PNG
- Generates interactive self-contained HTML/WebGL Three.js visualizer
"""

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def project_high_dim_spherical_to_3d(traces, D):
    """
    Fits PCA on all unit vectors u_q, u_w, and mem_u across time
    to find the 3 most significant geodesic directions in S^{D-1}.
    """
    all_u = []
    for t in traces:
        all_u.append(t["u_q"])
        all_u.append(t["u_w"])
        for u_slot in t["mem_u"]:
            all_u.append(u_slot)
    
    all_u = np.array(all_u) # [N_total, D]
    
    # Compute SVD / PCA on unit sphere vectors
    mean = np.mean(all_u, axis=0, keepdims=True)
    centered = all_u - mean
    _, _, Vt = np.linalg.svd(centered)
    proj_matrix = Vt[:3].T # [D, 3]

    def project_point(u_vec, r_val):
        u_arr = np.array(u_vec)
        # Project direction to 3D and re-normalize to unit sphere S^2
        u_3d = u_arr @ proj_matrix
        norm = np.linalg.norm(u_3d)
        if norm > 1e-6:
            u_3d = u_3d / norm
        else:
            u_3d = np.array([1.0, 0.0, 0.0])
        # Scale by radius r (infinite spherical coordinate)
        return u_3d * r_val

    return project_point

def render_matplotlib_3d(trace_file: str, output_image: str):
    with open(trace_file, "r") as f:
        data = json.load(f)

    traces = data["traces"]
    D = data["config"]["hyperspace_dim"]
    proj_fn = project_high_dim_spherical_to_3d(traces, D)

    fig = plt.figure(figsize=(16, 7), dpi=150)
    ax3d = fig.add_subplot(1, 2, 1, projection='3d')
    ax_attn = fig.add_subplot(1, 2, 2)

    # 1. Draw concentric hyperspherical shells
    u_sphere = np.linspace(0, 2 * np.pi, 30)
    v_sphere = np.linspace(0, np.pi, 20)
    radii = [1.0, 2.0, 3.0, 4.0]
    colors = ['#4a90e2', '#50e3c2', '#b8e986', '#f5a623']

    for r, col in zip(radii, colors):
        x = r * np.outer(np.cos(u_sphere), np.sin(v_sphere))
        y = r * np.outer(np.sin(u_sphere), np.sin(v_sphere))
        z = r * np.outer(np.ones(np.size(u_sphere)), np.cos(v_sphere))
        ax3d.plot_wireframe(x, y, z, color=col, alpha=0.12, linewidth=0.6)

    # 2. Extract trajectory points
    read_pts = np.array([proj_fn(t["u_q"], t["r_q"]) for t in traces])
    write_pts = np.array([proj_fn(t["u_w"], t["r_w"]) for t in traces])

    # Plot Read Trajectory
    ax3d.plot(read_pts[:, 0], read_pts[:, 1], read_pts[:, 2], 
              color='#00f0ff', linewidth=2.2, label='Read Head (u_q, r_q)', marker='o', markersize=4)
    # Plot Write Trajectory
    ax3d.plot(write_pts[:, 0], write_pts[:, 1], write_pts[:, 2], 
              color='#ff007f', linewidth=2.2, label='Write Head (u_w, r_w)', linestyle='--', marker='^', markersize=4)

    # Plot final Memory Slots
    final_trace = traces[-1]
    mem_pts = np.array([proj_fn(u, r) for u, r in zip(final_trace["mem_u"], final_trace["mem_r"])])
    weights = np.array(final_trace["read_weights"])
    sizes = 20 + 200 * (weights / (weights.max() + 1e-6))

    ax3d.scatter(mem_pts[:, 0], mem_pts[:, 1], mem_pts[:, 2],
                 c='#ffe600', s=sizes, edgecolors='black', label='Memory Slots (steered)', alpha=0.9)

    ax3d.set_title("Neural Controller in Infinite Spherical Hyperspace", fontsize=12, fontweight='bold', pad=12)
    ax3d.set_xlabel("X (S^2 x r)")
    ax3d.set_ylabel("Y (S^2 x r)")
    ax3d.set_zlabel("Z (S^2 x r)")
    ax3d.legend(loc='upper right', fontsize=9)
    ax3d.view_init(elev=25, azim=45)

    # Attention matrix across time
    attn_matrix = np.array([t["read_weights"] for t in traces])
    cax = ax_attn.imshow(attn_matrix, aspect='auto', cmap='plasma', interpolation='nearest')
    fig.colorbar(cax, ax=ax_attn, label='vMF-Radial Attention Weight')
    ax_attn.set_title("Read Head Addressing Distribution across Memory Slots", fontsize=12, fontweight='bold')
    ax_attn.set_xlabel("Memory Slot Index")
    ax_attn.set_ylabel("Time Step t")
    
    # Mark phases
    num_items = data["config"]["num_items"]
    distractor = data["config"]["distractor_steps"]
    ax_attn.axhline(num_items - 0.5, color='white', linestyle='--', linewidth=1.5, label='Store -> Deliberate')
    ax_attn.axhline(num_items + distractor - 0.5, color='cyan', linestyle='--', linewidth=1.5, label='Deliberate -> Recall')
    ax_attn.legend(loc='upper right', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_image, dpi=150)
    plt.close()
    print(f"[ISHMN] 3D Matplotlib trajectory plot saved to: {output_image}")

def generate_interactive_html(trace_file: str, output_html: str):
    with open(trace_file, "r") as f:
        data = json.load(f)

    traces_json_str = json.dumps(data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Infinite Spherical Hyperspace Neural Memory Visualizer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: #080b12;
            color: #e0e8f5;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
            overflow: hidden;
            height: 100vh;
            display: flex;
        }}
        #viewport {{
            flex: 1;
            height: 100%;
            position: relative;
        }}
        #canvas-container {{
            width: 100%;
            height: 100%;
        }}
        #sidebar {{
            width: 380px;
            background: rgba(14, 20, 36, 0.94);
            backdrop-filter: blur(12px);
            border-left: 1px solid rgba(80, 130, 240, 0.2);
            padding: 24px;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
            gap: 18px;
            box-shadow: -8px 0 24px rgba(0,0,0,0.5);
            z-index: 10;
        }}
        h1 {{
            font-size: 1.25rem;
            color: #58a6ff;
            text-transform: uppercase;
            letter-spacing: 1px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .badge {{
            font-size: 0.7rem;
            background: #1f6feb;
            color: #fff;
            padding: 2px 8px;
            border-radius: 12px;
        }}
        .panel {{
            background: rgba(22, 32, 56, 0.7);
            border: 1px solid rgba(80, 130, 240, 0.2);
            border-radius: 8px;
            padding: 14px;
        }}
        .panel-title {{
            font-size: 0.8rem;
            font-weight: 600;
            color: #8b949e;
            text-transform: uppercase;
            margin-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}
        .metric-card {{
            background: rgba(10, 15, 28, 0.6);
            border-radius: 6px;
            padding: 10px;
            border-left: 3px solid #58a6ff;
        }}
        .metric-val {{
            font-size: 1.2rem;
            font-weight: 700;
            color: #ffffff;
            font-family: monospace;
        }}
        .metric-lbl {{
            font-size: 0.72rem;
            color: #8b949e;
            margin-top: 2px;
        }}
        .controls {{
            display: flex;
            gap: 8px;
            align-items: center;
        }}
        button {{
            background: #238636;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-weight: 600;
            font-size: 0.85rem;
            transition: 0.2s;
        }}
        button:hover {{ background: #2ea043; }}
        button.secondary {{
            background: #21262d;
            border: 1px solid #30363d;
        }}
        button.secondary:hover {{ background: #30363d; }}
        input[type="range"] {{
            flex: 1;
            accent-color: #58a6ff;
            cursor: pointer;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 0.82rem;
            margin-bottom: 6px;
        }}
        .dot {{
            width: 12px;
            height: 12px;
            border-radius: 50%;
            display: inline-block;
        }}
        #slot-bars {{
            display: flex;
            gap: 3px;
            height: 48px;
            align-items: flex-end;
            margin-top: 6px;
            background: rgba(0,0,0,0.3);
            padding: 4px;
            border-radius: 4px;
        }}
        .slot-bar {{
            flex: 1;
            background: #388bfd;
            border-radius: 2px 2px 0 0;
            min-height: 2px;
            transition: height 0.15s ease;
        }}
        .hud-overlay {{
            position: absolute;
            top: 20px;
            left: 20px;
            pointer-events: none;
            background: rgba(10, 15, 30, 0.75);
            backdrop-filter: blur(8px);
            border: 1px solid rgba(80, 130, 240, 0.3);
            border-radius: 8px;
            padding: 14px 18px;
            font-size: 0.85rem;
            color: #e2e8f0;
            line-height: 1.5;
        }}
        .hud-title {{
            font-weight: 700;
            color: #60a5fa;
            margin-bottom: 4px;
        }}
    </style>
    <!-- Three.js & OrbitControls from CDN -->
    <script src="https://unpkg.com/three@0.160.0/build/three.min.js"></script>
    <script src="https://unpkg.com/three@0.160.0/examples/js/controls/OrbitControls.js"></script>
</head>
<body>
    <div id="viewport">
        <div id="canvas-container"></div>
        <div class="hud-overlay">
            <div class="hud-title">Infinite Spherical Hyperspace (S<sup>D-1</sup> &times; &reals;<sup>+</sup>)</div>
            <div>Phase: <b id="hud-phase" style="color: #4ade80;">Presentation</b></div>
            <div>Active Shell: <b id="hud-shell">r = 2.41</b></div>
            <div>Addressing Focus: <b id="hud-focus">&kappa; = 4.82</b></div>
            <div style="font-size: 0.75rem; color: #94a3b8; margin-top: 4px;">&bull; Left Drag: Orbit | Right Drag: Pan | Scroll: Zoom</div>
        </div>
    </div>

    <div id="sidebar">
        <div>
            <h1>Hyperspace Net <span class="badge">ISHMN</span></h1>
            <p style="font-size: 0.78rem; color: #8b949e; margin-top: 4px;">
                Riemannian Geodesic Controller & Infinite Concentric Shells
            </p>
        </div>

        <div class="panel">
            <div class="panel-title">Playback Control</div>
            <div class="controls" style="margin-bottom: 12px;">
                <button id="btn-play">Pause</button>
                <button id="btn-reset" class="secondary">Reset</button>
                <span id="step-display" style="font-family: monospace; font-size: 0.85rem;">Step 0/0</span>
            </div>
            <input type="range" id="time-slider" min="0" max="0" value="0">
        </div>

        <div class="panel">
            <div class="panel-title">Controller Heads Live Coordinates</div>
            <div class="metric-grid">
                <div class="metric-card" style="border-left-color: #00f0ff;">
                    <div class="metric-val" id="val-rq">0.00</div>
                    <div class="metric-lbl">Read Radius (r<sub>q</sub>)</div>
                </div>
                <div class="metric-card" style="border-left-color: #ff007f;">
                    <div class="metric-val" id="val-rw">0.00</div>
                    <div class="metric-lbl">Write Radius (r<sub>w</sub>)</div>
                </div>
                <div class="metric-card" style="border-left-color: #ffe600;">
                    <div class="metric-val" id="val-wgate">0.00</div>
                    <div class="metric-lbl">Write Gate (g<sub>w</sub>)</div>
                </div>
                <div class="metric-card" style="border-left-color: #a855f7;">
                    <div class="metric-val" id="val-slots">16</div>
                    <div class="metric-lbl">Memory Slots (N)</div>
                </div>
            </div>
        </div>

        <div class="panel">
            <div class="panel-title">vMF Attention Across Memory Slots</div>
            <div id="slot-bars"></div>
        </div>

        <div class="panel">
            <div class="panel-title">Hyperspace Topology Legend</div>
            <div class="legend-item">
                <span class="dot" style="background: #00f0ff; box-shadow: 0 0 8px #00f0ff;"></span>
                <span><b>Read Head Query</b> (Direction & Radius)</span>
            </div>
            <div class="legend-item">
                <span class="dot" style="background: #ff007f; box-shadow: 0 0 8px #ff007f;"></span>
                <span><b>Write Head</b> (Gated Store Vector)</span>
            </div>
            <div class="legend-item">
                <span class="dot" style="background: #ffe600;"></span>
                <span><b>Memory Slot</b> (Active Riemannian Node)</span>
            </div>
            <div class="legend-item">
                <span class="dot" style="background: rgba(80, 150, 255, 0.4); border: 1px solid #58a6ff;"></span>
                <span><b>Concentric Spherical Shells</b> (r &rarr; &infin;)</span>
            </div>
        </div>
    </div>

    <script>
        const simData = {traces_json_str};
        const traces = simData.traces;
        const config = simData.config;

        const totalSteps = traces.length;
        let currentStep = 0;
        let isPlaying = true;
        let lastFrameTime = 0;

        // UI elements
        const slider = document.getElementById('time-slider');
        slider.max = totalSteps - 1;
        const stepDisplay = document.getElementById('step-display');
        const btnPlay = document.getElementById('btn-play');
        const btnReset = document.getElementById('btn-reset');
        const hudPhase = document.getElementById('hud-phase');
        const hudShell = document.getElementById('hud-shell');
        const hudFocus = document.getElementById('hud-focus');
        const valRq = document.getElementById('val-rq');
        const valRw = document.getElementById('val-rw');
        const valWgate = document.getElementById('val-wgate');
        const slotBarsContainer = document.getElementById('slot-bars');

        // Init attention bars
        const numSlots = traces[0].mem_u.length;
        const barElements = [];
        for (let i = 0; i < numSlots; i++) {{
            const bar = document.createElement('div');
            bar.className = 'slot-bar';
            slotBarsContainer.appendChild(bar);
            barElements.push(bar);
        }}

        // Setup Three.js Scene
        const container = document.getElementById('canvas-container');
        const scene = new THREE.Scene();
        scene.fog = new THREE.FogExp2(0x080b12, 0.025);

        const camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 1000);
        camera.position.set(12, 9, 14);

        const renderer = new THREE.WebGLRenderer({{ antialias: true, alpha: true }});
        renderer.setSize(container.clientWidth, container.clientHeight);
        renderer.setPixelRatio(window.devicePixelRatio);
        container.appendChild(renderer.domElement);

        const controls = new THREE.OrbitControls(camera, renderer.domElement);
        controls.enableDamping = true;
        controls.dampingFactor = 0.05;

        // Lights
        const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
        scene.add(ambientLight);

        const dirLight = new THREE.DirectionalLight(0x80b0ff, 1.2);
        dirLight.position.set(10, 20, 10);
        scene.add(dirLight);

        // Concentric Spherical Shells
        const shells = [1.5, 3.0, 4.5, 6.0];
        const shellMeshes = [];
        shells.forEach((r, idx) => {{
            const geom = new THREE.SphereGeometry(r, 28, 20);
            const mat = new THREE.MeshBasicMaterial({{
                color: [0x388bfd, 0x2ea043, 0xd29922, 0xa371f7][idx % 4],
                wireframe: true,
                transparent: true,
                opacity: 0.12 - idx * 0.015
            }});
            const shellMesh = new THREE.Mesh(geom, mat);
            scene.add(shellMesh);
            shellMeshes.push(shellMesh);
        }});

        // Dynamic 3D Objects
        // Read Head Pointer
        const readHeadGeom = new THREE.SphereGeometry(0.28, 16, 16);
        const readHeadMat = new THREE.MeshStandardMaterial({{
            color: 0x00f0ff,
            emissive: 0x00a0ff,
            emissiveIntensity: 0.8,
            roughness: 0.2
        }});
        const readHeadMesh = new THREE.Mesh(readHeadGeom, readHeadMat);
        scene.add(readHeadMesh);

        // Write Head Pointer
        const writeHeadGeom = new THREE.OctahedronGeometry(0.32);
        const writeHeadMat = new THREE.MeshStandardMaterial({{
            color: 0xff007f,
            emissive: 0xff0055,
            emissiveIntensity: 0.8,
            roughness: 0.2
        }});
        const writeHeadMesh = new THREE.Mesh(writeHeadGeom, writeHeadMat);
        scene.add(writeHeadMesh);

        // Memory Slots Meshes
        const slotMeshes = [];
        for (let i = 0; i < numSlots; i++) {{
            const geom = new THREE.SphereGeometry(0.18, 12, 12);
            const mat = new THREE.MeshStandardMaterial({{
                color: 0xffe600,
                emissive: 0xffaa00,
                emissiveIntensity: 0.3
            }});
            const mesh = new THREE.Mesh(geom, mat);
            scene.add(mesh);
            slotMeshes.push(mesh);
        }}

        // Read Trajectory Ribbon
        const maxHistory = 40;
        const readHistoryPts = [];
        const readLineGeom = new THREE.BufferGeometry();
        const readLineMat = new THREE.LineBasicMaterial({{ color: 0x00f0ff, transparent: true, opacity: 0.8 }});
        const readLine = new THREE.Line(readLineGeom, readLineMat);
        scene.add(readLine);

        // Project D-dim direction + radius to 3D Cartesian
        function projectTo3D(u, r) {{
            // Simple robust projection: pick first 3 dimensions and re-normalize to unit 3D sphere
            let x = u[0] || 0;
            let y = u[1] || 0;
            let z = u[2] || 0;
            const norm = Math.hypot(x, y, z) || 1.0;
            return new THREE.Vector3((x / norm) * r, (y / norm) * r, (z / norm) * r);
        }}

        function updateStep(step) {{
            currentStep = Math.max(0, Math.min(totalSteps - 1, step));
            slider.value = currentStep;
            stepDisplay.innerText = `Step ${{currentStep + 1}}/${{totalSteps}}`;

            const tData = traces[currentStep];

            // Update Head Positions
            const readPos = projectTo3D(tData.u_q, tData.r_q);
            const writePos = projectTo3D(tData.u_w, tData.r_w);

            readHeadMesh.position.copy(readPos);
            writeHeadMesh.position.copy(writePos);
            writeHeadMesh.scale.setScalar(0.5 + tData.write_gate * 0.8);

            // Trajectory Trail
            readHistoryPts.push(readPos.clone());
            if (readHistoryPts.length > maxHistory) readHistoryPts.shift();
            readLineGeom.setFromPoints(readHistoryPts);

            // Update Memory Slots
            for (let i = 0; i < numSlots; i++) {{
                const u_i = tData.mem_u[i];
                const r_i = tData.mem_r[i];
                const pos = projectTo3D(u_i, r_i);
                slotMeshes[i].position.copy(pos);

                const attn = tData.read_weights[i] || 0;
                const scale = 0.8 + attn * 3.5;
                slotMeshes[i].scale.setScalar(scale);
                slotMeshes[i].material.emissiveIntensity = 0.2 + attn * 1.5;

                // Update HTML Bar
                barElements[i].style.height = `${{Math.min(100, attn * 250)}}%`;
                barElements[i].style.background = attn > 0.15 ? '#00f0ff' : '#388bfd';
            }}

            // Update Telemetry Panel
            valRq.innerText = tData.r_q.toFixed(2);
            valRw.innerText = tData.r_w.toFixed(2);
            valWgate.innerText = tData.write_gate.toFixed(2);

            // Phase tracking
            const numItems = config.num_items;
            const distractor = config.distractor_steps;
            if (currentStep < numItems) {{
                hudPhase.innerText = "1. Item Presentation (Writing)";
                hudPhase.style.color = "#ff007f";
            }} else if (currentStep < numItems + distractor) {{
                hudPhase.innerText = "2. Deliberation & Hyperspace Holding";
                hudPhase.style.color = "#eab308";
            }} else {{
                hudPhase.innerText = "3. Associative Recall & Reading";
                hudPhase.style.color = "#00f0ff";
            }}
            hudShell.innerText = `r = ${{tData.r_q.toFixed(2)}}`;
            hudFocus.innerText = `gate = ${{tData.write_gate.toFixed(2)}}`;
        }}

        // Controls event listeners
        slider.addEventListener('input', (e) => {{
            readHistoryPts.length = 0; // reset trail on jump
            updateStep(parseInt(e.target.value));
        }});

        btnPlay.addEventListener('click', () => {{
            isPlaying = !isPlaying;
            btnPlay.innerText = isPlaying ? "Pause" : "Play";
        }});

        btnReset.addEventListener('click', () => {{
            readHistoryPts.length = 0;
            updateStep(0);
        }});

        window.addEventListener('resize', () => {{
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
        }});

        // Animation Loop
        function animate(timestamp) {{
            requestAnimationFrame(animate);

            if (isPlaying && timestamp - lastFrameTime > 280) {{
                lastFrameTime = timestamp;
                let nextStep = currentStep + 1;
                if (nextStep >= totalSteps) nextStep = 0;
                updateStep(nextStep);
            }}

            // Slowly rotate scene for dynamic perspective
            scene.rotation.y += 0.0015;

            controls.update();
            renderer.render(scene, camera);
        }}

        updateStep(0);
        requestAnimationFrame(animate);
    </script>
</body>
</html>
"""

    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"[ISHMN] Interactive 3D WebGL visualizer generated at: {output_html}")

if __name__ == "__main__":
    trace_path = "artifacts/traces.json"
    if os.path.exists(trace_path):
        render_matplotlib_3d(trace_path, "artifacts/spherical_memory_trajectory.png")
        generate_interactive_html(trace_path, "artifacts/spherical_memory_visualizer.html")
    else:
        print(f"Trace file not found at {trace_path}. Run demo_train.py first.")
