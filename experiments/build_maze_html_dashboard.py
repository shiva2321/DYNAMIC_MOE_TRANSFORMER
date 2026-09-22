"""
Builds the standalone interactive HTML live maze visualizer.
Embeds artifacts/maze_visualizer_data.json directly into artifacts/live_maze_visualizer.html.
"""

import json

def generate_html():
    with open("artifacts/maze_visualizer_data.json", "r") as f:
        maze_data = json.load(f)

    json_str = json.dumps(maze_data)

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Universal Cognitive Engine | Live Neural Maze Solver</title>
    <style>
        :root {{
            --bg: #07090e;
            --card-bg: #0e131f;
            --card-border: #1e293b;
            --text: #f8fafc;
            --text-dim: #94a3b8;
            --accent-cyan: #00f0ff;
            --accent-green: #10b981;
            --accent-yellow: #fbbf24;
            --accent-red: #f43f5e;
            --accent-purple: #a855f7;
            --wall-color: #1e293b;
            --wall-border: #334155;
            --cell-bg: #0b0f19;
            --trail-color: rgba(0, 240, 255, 0.45);
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        }}

        body {{
            background-color: var(--bg);
            color: var(--text);
            min-height: 100vh;
            padding: 24px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }}

        .container {{
            width: 100%;
            max-width: 1280px;
            display: flex;
            flex-direction: column;
            gap: 20px;
        }}

        header {{
            background: linear-gradient(135deg, rgba(14, 19, 31, 0.9), rgba(15, 23, 42, 0.6));
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 20px 28px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(12px);
        }}

        .title-block h1 {{
            font-size: 22px;
            font-weight: 700;
            background: linear-gradient(90deg, #00f0ff, #a855f7);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        }}

        .title-block p {{
            font-size: 13px;
            color: var(--text-dim);
        }}

        .badge-bar {{
            display: flex;
            gap: 10px;
        }}

        .badge {{
            padding: 6px 14px;
            border-radius: 9999px;
            font-size: 11px;
            font-weight: 600;
            letter-spacing: 0.5px;
            text-transform: uppercase;
        }}

        .badge-live {{
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.4);
            color: var(--accent-green);
        }}

        .badge-algo {{
            background: rgba(168, 85, 247, 0.15);
            border: 1px solid rgba(168, 85, 247, 0.4);
            color: var(--accent-purple);
        }}

        /* Selector Deck */
        .selector-deck {{
            display: flex;
            gap: 12px;
            align-items: center;
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 12px 18px;
        }}

        .selector-deck label {{
            font-size: 13px;
            font-weight: 600;
            color: var(--text-dim);
        }}

        select {{
            background: #141b2d;
            color: var(--text);
            border: 1px solid var(--card-border);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 500;
            outline: none;
            cursor: pointer;
            transition: all 0.2s;
        }}

        select:focus {{
            border-color: var(--accent-cyan);
            box-shadow: 0 0 10px rgba(0, 240, 255, 0.2);
        }}

        .cat-pills {{
            display: flex;
            gap: 8px;
            margin-left: auto;
        }}

        .pill-btn {{
            background: transparent;
            border: 1px solid var(--card-border);
            color: var(--text-dim);
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .pill-btn.active, .pill-btn:hover {{
            background: rgba(0, 240, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}

        /* Main Workspace Grid */
        .workspace {{
            display: grid;
            grid-template-columns: 1.15fr 0.85fr;
            gap: 20px;
        }}

        .card {{
            background: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 18px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        }}

        /* Maze Stage */
        .stage-card {{
            align-items: center;
        }}

        .stage-header {{
            width: 100%;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .stage-title {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text);
        }}

        .step-counter {{
            font-size: 13px;
            font-weight: 600;
            color: var(--accent-cyan);
            background: rgba(0, 240, 255, 0.1);
            padding: 4px 10px;
            border-radius: 6px;
        }}

        #mazeCanvas {{
            background: var(--cell-bg);
            border: 2px solid var(--card-border);
            border-radius: 12px;
            box-shadow: inset 0 0 20px rgba(0, 0, 0, 0.8), 0 0 25px rgba(0, 240, 255, 0.05);
        }}

        /* Control Deck */
        .controls {{
            width: 100%;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
            background: #121826;
            border: 1px solid var(--card-border);
            padding: 12px 18px;
            border-radius: 12px;
        }}

        .btn {{
            background: #1a2236;
            color: var(--text);
            border: 1px solid var(--card-border);
            padding: 8px 14px;
            border-radius: 8px;
            font-size: 13px;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.2s;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }}

        .btn:hover {{
            background: #253352;
            border-color: var(--accent-cyan);
            color: #fff;
        }}

        .btn-play {{
            background: rgba(0, 240, 255, 0.15);
            border-color: var(--accent-cyan);
            color: var(--accent-cyan);
        }}

        .btn-play:hover {{
            background: var(--accent-cyan);
            color: #000;
        }}

        .speed-control {{
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 12px;
            color: var(--text-dim);
            margin-left: 12px;
        }}

        input[type="range"] {{
            accent-color: var(--accent-cyan);
            cursor: pointer;
        }}

        /* Telemetry Deck */
        .telemetry-card {{
            gap: 16px;
        }}

        .section-title {{
            font-size: 13px;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            color: var(--text-dim);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .section-title::before {{
            content: "";
            width: 4px;
            height: 14px;
            background: var(--accent-cyan);
            border-radius: 2px;
        }}

        /* Deliberation Banner */
        .delib-box {{
            background: #141b2c;
            border: 1px solid var(--card-border);
            border-radius: 10px;
            padding: 14px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .delib-mode {{
            font-size: 14px;
            font-weight: 700;
        }}

        .mode-sys1 {{
            color: var(--accent-cyan);
        }}

        .mode-sys2 {{
            color: var(--accent-yellow);
        }}

        /* Action Bars */
        .action-bars {{
            display: flex;
            flex-direction: column;
            gap: 8px;
        }}

        .action-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 12px;
            font-weight: 600;
        }}

        .action-label {{
            width: 65px;
            color: var(--text-dim);
        }}

        .bar-container {{
            flex: 1;
            height: 14px;
            background: #141a29;
            border-radius: 999px;
            overflow: hidden;
            position: relative;
        }}

        .bar-fill {{
            height: 100%;
            background: linear-gradient(90deg, #00f0ff, #a855f7);
            border-radius: 999px;
            width: 0%;
            transition: width 0.15s ease;
        }}

        .action-row.winning .action-label {{
            color: var(--accent-cyan);
        }}

        .action-row.winning .bar-fill {{
            background: linear-gradient(90deg, #10b981, #00f0ff);
            box-shadow: 0 0 10px rgba(16, 185, 129, 0.4);
        }}

        .action-val {{
            width: 38px;
            text-align: right;
            font-size: 11px;
            color: var(--text);
        }}

        /* MoE Specialists */
        .moe-grid {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
        }}

        .moe-tile {{
            background: #141b2c;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 10px;
            text-align: center;
        }}

        .moe-tile-title {{
            font-size: 11px;
            color: var(--text-dim);
            margin-bottom: 4px;
        }}

        .moe-tile-val {{
            font-size: 14px;
            font-weight: 700;
            color: var(--accent-purple);
        }}

        /* Metrics summary */
        .metrics-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 10px;
        }}

        .metric-card {{
            background: #121826;
            border: 1px solid var(--card-border);
            border-radius: 8px;
            padding: 10px 14px;
        }}

        .metric-lbl {{
            font-size: 11px;
            color: var(--text-dim);
        }}

        .metric-val {{
            font-size: 16px;
            font-weight: 700;
            color: var(--text);
            margin-top: 2px;
        }}

        /* Footer */
        footer {{
            text-align: center;
            font-size: 12px;
            color: var(--text-dim);
            margin-top: 10px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <header>
            <div class="title-block">
                <h1>Universal Cognitive Engine &bull; Live Spatial Solver</h1>
                <p>Pure Neural Navigation on Riemannian Hypersphere (Zero Hardcoded Pathfinding Algorithms)</p>
            </div>
            <div class="badge-bar">
                <div class="badge badge-live">&bull; Live Neural Playback</div>
                <div class="badge badge-algo">S^7 x R+ Manifold</div>
            </div>
        </header>

        <!-- Selector Deck -->
        <div class="selector-deck">
            <label for="mazeSelect">Select Layout:</label>
            <select id="mazeSelect"></select>

            <div class="cat-pills">
                <button class="pill-btn active" data-cat="all">All</button>
                <button class="pill-btn" data-cat="Seen (Training)">Seen</button>
                <button class="pill-btn" data-cat="Unseen (In-Distribution)">Unseen</button>
                <button class="pill-btn" data-cat="Novel (Out-of-Distribution)">Novel OOD</button>
            </div>
        </div>

        <!-- Main Workspace -->
        <div class="workspace">
            <!-- Stage Left: Canvas & Playback -->
            <div class="card stage-card">
                <div class="stage-header">
                    <span class="stage-title" id="stageTitle">Loading Maze...</span>
                    <span class="step-counter" id="stepCounter">Step 0 / 0</span>
                </div>

                <canvas id="mazeCanvas" width="460" height="460"></canvas>

                <div class="controls">
                    <button class="btn" id="btnReset">&#8634; Reset</button>
                    <button class="btn" id="btnPrev">&#9664; Step</button>
                    <button class="btn btn-play" id="btnPlay">&#9654; Play</button>
                    <button class="btn" id="btnNext">Step &#9654;</button>

                    <div class="speed-control">
                        <span>Speed:</span>
                        <input type="range" id="speedSlider" min="50" max="600" value="200" step="25">
                    </div>
                </div>
            </div>

            <!-- Stage Right: Cognitive Telemetry -->
            <div class="card telemetry-card">
                <!-- Deliberation Mode -->
                <div>
                    <div class="section-title">Cognitive Mode & Deliberation</div>
                    <div class="delib-box">
                        <span class="delib-mode mode-sys1" id="delibModeText">System 1 (Corridor Flow)</span>
                        <span style="font-size: 12px; color: var(--text-dim);" id="branchText">Open branches: 2</span>
                    </div>
                </div>

                <!-- Action Logits -->
                <div>
                    <div class="section-title">Action Policy Distribution</div>
                    <div class="action-bars">
                        <div class="action-row" id="row-0">
                            <span class="action-label">&#9650; UP</span>
                            <div class="bar-container"><div class="bar-fill" id="bar-0"></div></div>
                            <span class="action-val" id="val-0">0%</span>
                        </div>
                        <div class="action-row" id="row-1">
                            <span class="action-label">&#9660; DOWN</span>
                            <div class="bar-container"><div class="bar-fill" id="bar-1"></div></div>
                            <span class="action-val" id="val-1">0%</span>
                        </div>
                        <div class="action-row" id="row-2">
                            <span class="action-label">&#9668; LEFT</span>
                            <div class="bar-container"><div class="bar-fill" id="bar-2"></div></div>
                            <span class="action-val" id="val-2">0%</span>
                        </div>
                        <div class="action-row" id="row-3">
                            <span class="action-label">&#9658; RIGHT</span>
                            <div class="bar-container"><div class="bar-fill" id="bar-3"></div></div>
                            <span class="action-val" id="val-3">0%</span>
                        </div>
                    </div>
                </div>

                <!-- MoE Specialists -->
                <div>
                    <div class="section-title">Spherical MoE Specialist Weights</div>
                    <div class="moe-grid">
                        <div class="moe-tile">
                            <div class="moe-tile-title">Expert 0</div>
                            <div class="moe-tile-val" id="moe-0">0.00</div>
                        </div>
                        <div class="moe-tile">
                            <div class="moe-tile-title">Expert 1</div>
                            <div class="moe-tile-val" id="moe-1">0.00</div>
                        </div>
                        <div class="moe-tile">
                            <div class="moe-tile-title">Expert 2</div>
                            <div class="moe-tile-val" id="moe-2">0.00</div>
                        </div>
                        <div class="moe-tile">
                            <div class="moe-tile-title">Expert 3</div>
                            <div class="moe-tile-val" id="moe-3">0.00</div>
                        </div>
                    </div>
                </div>

                <!-- Diagnostics -->
                <div>
                    <div class="section-title">Navigation Diagnostics</div>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-lbl">Status</div>
                            <div class="metric-val" id="diagStatus" style="color: var(--accent-cyan);">In Progress</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-lbl">Path Efficiency</div>
                            <div class="metric-val" id="diagEff">1.00x</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-lbl">Steps (Optimal)</div>
                            <div class="metric-val" id="diagSteps">0 (0)</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-lbl">Wall Collisions</div>
                            <div class="metric-val" id="diagCrashes">0</div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <footer>
            Universal Cognitive Engine &bull; Infinite Spherical Hyperspace Memory Network (ISHMN) &bull; Pure Geodesic Intelligence
        </footer>
    </div>

    <!-- Live Script -->
    <script>
        const MAZE_DATA = {json_str};

        let currentMazeIndex = 0;
        let currentStep = 0;
        let isPlaying = false;
        let playInterval = null;

        const canvas = document.getElementById('mazeCanvas');
        const ctx = canvas.getContext('2d');
        const select = document.getElementById('mazeSelect');
        const stageTitle = document.getElementById('stageTitle');
        const stepCounter = document.getElementById('stepCounter');
        const btnPlay = document.getElementById('btnPlay');
        const btnReset = document.getElementById('btnReset');
        const btnPrev = document.getElementById('btnPrev');
        const btnNext = document.getElementById('btnNext');
        const speedSlider = document.getElementById('speedSlider');

        // Populate Dropdown
        function populateSelect(filterCat = "all") {{
            select.innerHTML = "";
            MAZE_DATA.forEach((m, idx) => {{
                if (filterCat === "all" || m.category === filterCat) {{
                    const opt = document.createElement('option');
                    opt.value = idx;
                    opt.textContent = `[${{m.category}}] ${{m.label}}`;
                    select.appendChild(opt);
                }}
            }});
            select.value = currentMazeIndex;
        }}

        function loadMaze(idx) {{
            currentMazeIndex = idx;
            currentStep = 0;
            pause();
            updateUI();
        }}

        function updateUI() {{
            const maze = MAZE_DATA[currentMazeIndex];
            stageTitle.textContent = `${{maze.label}} (${{maze.grid_size}}x${{maze.grid_size}} ${{maze.category}})`;

            const totalFrames = maze.frames.length;
            stepCounter.textContent = `Step ${{currentStep}} / ${{totalFrames - 1}}`;

            const frame = maze.frames[currentStep];

            // Render Canvas
            renderCanvas(maze, frame);

            // Render Telemetry
            renderTelemetry(maze, frame);
        }}

        function renderCanvas(maze, frame) {{
            const N = maze.grid_size;
            const size = canvas.width;
            const cellSize = size / N;

            ctx.clearRect(0, 0, size, size);

            // Draw Background grid
            for (let r = 0; r < N; r++) {{
                for (let c = 0; c < N; c++) {{
                    ctx.strokeStyle = "#161f30";
                    ctx.lineWidth = 1;
                    ctx.strokeRect(c * cellSize, r * cellSize, cellSize, cellSize);

                    if (maze.walls[r][c] === 1.0) {{
                        // Draw Wall
                        ctx.fillStyle = "#1e293b";
                        ctx.fillRect(c * cellSize + 2, r * cellSize + 2, cellSize - 4, cellSize - 4);

                        ctx.strokeStyle = "#334155";
                        ctx.lineWidth = 2;
                        ctx.strokeRect(c * cellSize + 2, r * cellSize + 2, cellSize - 4, cellSize - 4);
                    }}
                }}
            }}

            // Draw Footprint Trail
            const trailSoFar = maze.path_history.slice(0, currentStep + 1);
            if (trailSoFar.length > 1) {{
                for (let i = 0; i < trailSoFar.length - 1; i++) {{
                    const [r, c] = trailSoFar[i];
                    ctx.fillStyle = "rgba(0, 240, 255, 0.4)";
                    ctx.beginPath();
                    ctx.arc(c * cellSize + cellSize / 2, r * cellSize + cellSize / 2, cellSize * 0.14, 0, Math.PI * 2);
                    ctx.fill();
                }}
            }}

            // Draw Goal Star
            const goal = maze.optimal_path[maze.optimal_path.length - 1];
            const gx = goal[1] * cellSize + cellSize / 2;
            const gy = goal[0] * cellSize + cellSize / 2;
            ctx.fillStyle = "#fbbf24";
            ctx.shadowColor = "#fbbf24";
            ctx.shadowBlur = 12;
            ctx.font = `${{cellSize * 0.55}}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("★", gx, gy);
            ctx.shadowBlur = 0;

            // Draw Agent
            const curr = frame.curr;
            const ax = curr[1] * cellSize + cellSize / 2;
            const ay = curr[0] * cellSize + cellSize / 2;

            ctx.fillStyle = "#10b981";
            ctx.shadowColor = "#10b981";
            ctx.shadowBlur = 15;
            ctx.beginPath();
            ctx.arc(ax, ay, cellSize * 0.32, 0, Math.PI * 2);
            ctx.fill();

            ctx.fillStyle = "#ffffff";
            ctx.font = `${{cellSize * 0.38}}px sans-serif`;
            ctx.textAlign = "center";
            ctx.textBaseline = "middle";
            ctx.fillText("🤖", ax, ay);
            ctx.shadowBlur = 0;
        }}

        function renderTelemetry(maze, frame) {{
            // Deliberation
            const delibEl = document.getElementById('delibModeText');
            delibEl.textContent = frame.system_mode;
            if (frame.system_mode.includes('System 2')) {{
                delibEl.className = 'delib-mode mode-sys2';
            }} else {{
                delibEl.className = 'delib-mode mode-sys1';
            }}

            // Action bars
            const probs = frame.action_probs;
            let maxP = -1;
            let maxIdx = -1;
            for (let i = 0; i < 4; i++) {{
                if (probs[i] > maxP) {{
                    maxP = probs[i];
                    maxIdx = i;
                }}
            }}

            for (let i = 0; i < 4; i++) {{
                const p = probs[i];
                const pct = Math.round(p * 100);
                document.getElementById(`bar-${{i}}`).style.width = `${{pct}}%`;
                document.getElementById(`val-${{i}}`).textContent = `${{pct}}%`;

                const row = document.getElementById(`row-${{i}}`);
                if (i === maxIdx && !frame.reached_goal) {{
                    row.classList.add('winning');
                }} else {{
                    row.classList.remove('winning');
                }}
            }}

            // MoE weights
            const moe = frame.moe_weights;
            for (let i = 0; i < 4; i++) {{
                document.getElementById(`moe-${{i}}`).textContent = (moe[i] !== undefined) ? moe[i].toFixed(2) : "0.00";
            }}

            // Diagnostics
            const diagStatus = document.getElementById('diagStatus');
            if (frame.reached_goal) {{
                diagStatus.textContent = "Goal Reached!";
                diagStatus.style.color = "#10b981";
            }} else {{
                diagStatus.textContent = "Navigating...";
                diagStatus.style.color = "#00f0ff";
            }}

            const eff = (maze.steps_taken / maze.optimal_steps).toFixed(2);
            document.getElementById('diagEff').textContent = `${{eff}}x optimal`;
            document.getElementById('diagSteps').textContent = `${{currentStep}} (Opt: ${{maze.optimal_steps}})`;
            document.getElementById('diagCrashes').textContent = maze.crashes;
        }}

        function play() {{
            if (isPlaying) return;
            isPlaying = true;
            btnPlay.innerHTML = "&#10074;&#10074; Pause";
            const speed = 650 - parseInt(speedSlider.value);

            playInterval = setInterval(() => {{
                const maze = MAZE_DATA[currentMazeIndex];
                if (currentStep < maze.frames.length - 1) {{
                    currentStep++;
                    updateUI();
                }} else {{
                    pause();
                }}
            }}, speed);
        }}

        function pause() {{
            isPlaying = false;
            btnPlay.innerHTML = "&#9654; Play";
            if (playInterval) {{
                clearInterval(playInterval);
                playInterval = null;
            }}
        }}

        btnPlay.addEventListener('click', () => {{
            if (isPlaying) pause();
            else play();
        }});

        btnReset.addEventListener('click', () => {{
            pause();
            currentStep = 0;
            updateUI();
        }});

        btnPrev.addEventListener('click', () => {{
            pause();
            if (currentStep > 0) {{
                currentStep--;
                updateUI();
            }}
        }});

        btnNext.addEventListener('click', () => {{
            pause();
            const maze = MAZE_DATA[currentMazeIndex];
            if (currentStep < maze.frames.length - 1) {{
                currentStep++;
                updateUI();
            }}
        }});

        select.addEventListener('change', (e) => {{
            loadMaze(parseInt(e.target.value));
        }});

        // Filter pills
        document.querySelectorAll('.pill-btn').forEach(btn => {{
            btn.addEventListener('click', () => {{
                document.querySelectorAll('.pill-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                populateSelect(btn.dataset.cat);
                loadMaze(parseInt(select.value));
            }});
        }});

        // Initialize
        populateSelect("all");
        loadMaze(0);
    </script>
</body>
</html>
"""

    with open("artifacts/live_maze_visualizer.html", "w", encoding="utf-8") as f:
        f.write(html_content)

    print("[+] Generated standalone interactive HTML dashboard at artifacts/live_maze_visualizer.html!")

if __name__ == "__main__":
    generate_html()
