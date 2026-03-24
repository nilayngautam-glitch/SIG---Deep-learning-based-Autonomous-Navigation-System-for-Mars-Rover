import streamlit as st
import numpy as np
import cv2
from io import BytesIO
from PIL import Image
import base64
from utils.ml_engine import (
    TF_AVAILABLE, extract_patches, preprocess_patches,
    build_terrain_grid, build_safety_grid, compute_terrain_report,
    CLASS_NAMES, PATCH_SIZE, IMG_SIZE
)

# ── Color map for each terrain class (kept as before) ──
TERRAIN_COLORS = {
    "Bedrock": "#10B981",   # Green
    "Sand":    "#F59E0B",   # Yellow
    "Rocks":   "#EF4444",   # Red
    "Pebbles": "#7C3AED",   # Purple
}

def img_to_base64(img_rgb):
    buffer = BytesIO()
    Image.fromarray(img_rgb).save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def run_terrain_prediction(img_rgb, model):
    """Run the full ML pipeline: patches → predict → terrain report + safety score"""
    # Step 1: Extract patches from image
    patches, positions = extract_patches(img_rgb, PATCH_SIZE)
    if len(patches) == 0:
        return None, None

    # Step 2: Preprocess patches for MobileNetV2
    processed = preprocess_patches(patches, IMG_SIZE)

    # Step 3: Model prediction
    predictions = model.predict(processed, verbose=0)
    pred_classes = np.argmax(predictions, axis=1)

    # Step 4: Build grids
    terrain_grid = build_terrain_grid(img_rgb.shape, PATCH_SIZE, pred_classes)
    safety_grid = build_safety_grid(terrain_grid)

    # Step 5: Compute outputs
    safety_score = int(np.mean(safety_grid) * 100)  # Average safety as percentage
    terrain_report = compute_terrain_report(terrain_grid)

    return safety_score, terrain_report, terrain_grid, safety_grid


def render():
    """This function is called by app.py when user clicks 'Analysis' in sidebar"""
    
    # ── Step 1: Page Title ──
    st.markdown("""
    <div style="margin-bottom: 0.5rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0;">🔬 Terrain Analysis</h1>
        <p style="color: #6B6B8A; font-size: 0.9rem; margin-top: 4px;">
            AI-powered terrain classification and navigation
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Step 2 - Image Upload
    uploaded_image = st.file_uploader(
        "Upload Mars terrain image",
        type=["jpg", "jpeg", "png", "bmp", "tiff"],
        label_visibility="collapsed"
    )

    if uploaded_image is not None:
        file_bytes = np.asarray(bytearray(uploaded_image.read()), dtype=np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        st.session_state['img_rgb'] = img_rgb

        # ═══════════════════════════════════════
        #  ML PREDICTION — real model data
        # ═══════════════════════════════════════
        model = st.session_state.get('model', None)
        
        terrain_grid = None
        safety_grid = None

        if model is not None and TF_AVAILABLE:
            # Run the actual ML pipeline
            with st.spinner("🧠 Analyzing terrain..."):
                safety_score, terrain_report, terrain_grid, safety_grid = run_terrain_prediction(img_rgb, model)

            if safety_score is None:
                st.warning("⚠ Image too small to extract patches. Try a larger image.")
                return

            # Build terrain_types dict from real predictions
            terrain_types = {}
            for entry in terrain_report:
                name = entry["terrain"]
                pct = round(entry["percentage"], 1)
                color = TERRAIN_COLORS.get(name, "#7C3AED")
                terrain_types[name] = {"pct": pct, "color": color}

            # Generate description based on score
            if safety_score >= 70:
                desc_text = "Current terrain profile exhibits standard risk parameters. Proceed with Alpha sequence."
            elif safety_score >= 40:
                desc_text = "Mixed terrain detected. Exercise caution and verify path integrity before proceeding."
            else:
                desc_text = "Hazardous terrain detected. Recommend rerouting or manual inspection before traversal."

        else:
            # No model loaded — show warning
            st.warning("⚠ **No model loaded.** Load a .keras model in the sidebar to get real predictions.")
            safety_score = 0
            terrain_types = {}
            desc_text = "No model loaded. Safety analysis unavailable."

        # ── Gauge color/status based on score ──
        if safety_score >= 70:
            gauge_color = "#10B981"
            status_text = "HIGHLY NAVIGABLE"
        elif safety_score >= 40:
            gauge_color = "#F59E0B"
            status_text = "MODERATE RISK"
        else:
            gauge_color = "#EF4444"
            status_text = "HIGH RISK"

        # SVG circle math
        radius = 70
        circumference = 2 * 3.14159 * radius
        dash_offset = circumference * (1 - safety_score / 100)

        # ── Image data ──
        h, w = img_rgb.shape[:2]
        b64 = img_to_base64(img_rgb)

        # ═══════════════════════════════════════
        #  MAIN LAYOUT: Image (left) | Info Panel (right)
        # ═══════════════════════════════════════
        col_img, col_info = st.columns([2, 1])

        # ── LEFT COLUMN: Fancy Image Viewer ──
        with col_img:
            if safety_grid is not None and terrain_grid is not None:
                tab1, tab2, tab3 = st.tabs(["Raw Telemetry", "Safety Heatmap", "Terrain Classes"])
                
                with tab1:
                    st.markdown(f"""
                    <div class="image-viewer" style="position: relative; min-height: 300px; margin-top:0.5rem;">
                        <img src="{b64}" style="width:100%; display:block; border-radius:16px;">
                        <div class="corner-tl"></div><div class="corner-tr"></div>
                        <div class="corner-bl"></div><div class="corner-br"></div>
                        <div class="crosshair"></div>
                        <div class="image-overlay-info" style="top:12px; left:12px; bottom:auto;">
                            <span class="live-badge">● LIVE IMAGING TX</span><br>CAM-ID: PERSEVERANCE_FRONT_01
                        </div>
                        <div style="position:absolute; top:12px; right:12px; display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
                            <div class="disable-overlay-btn">🔗 Disable Overlay</div>
                            <div class="coord-badge" style="position:static;">LAT: -14.5684<br>LON: 175.4721</div>
                        </div>
                        <div style="position:absolute; bottom:12px; left:12px; font-family:'Orbitron',sans-serif; font-size:0.65rem; color:rgba(255,255,255,0.6);">
                            RES: {w}×{h} @ 60FPS<br>HDR: ENABLED | T-STAMP: 1044321
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with tab2:
                    from utils.plotting import plot_overlay
                    fig_safe = plot_overlay(img_rgb, safety_grid, cmap="RdYlGn", title="", alpha=0.6)
                    st.pyplot(fig_safe)
                    
                with tab3:
                    from utils.plotting import plot_overlay
                    fig_terr = plot_overlay(img_rgb, terrain_grid, cmap="jet", title="", alpha=0.55)
                    st.pyplot(fig_terr)
            else:
                st.markdown(f"""
                <div class="image-viewer" style="position: relative; min-height: 300px;">
                    <img src="{b64}" style="width:100%; display:block; border-radius:16px;">
                    <div class="corner-tl"></div><div class="corner-tr"></div>
                    <div class="corner-bl"></div><div class="corner-br"></div>
                    <div class="crosshair"></div>
                    <div class="image-overlay-info" style="top:12px; left:12px; bottom:auto;">
                        <span class="live-badge">● LIVE IMAGING TX</span><br>CAM-ID: PERSEVERANCE_FRONT_01
                    </div>
                    <div style="position:absolute; top:12px; right:12px; display:flex; flex-direction:column; align-items:flex-end; gap:6px;">
                        <div class="disable-overlay-btn">🔗 Disable Overlay</div>
                        <div class="coord-badge" style="position:static;">LAT: -14.5684<br>LON: 175.4721</div>
                    </div>
                    <div style="position:absolute; bottom:12px; left:12px; font-family:'Orbitron',sans-serif; font-size:0.65rem; color:rgba(255,255,255,0.6);">
                        RES: {w}×{h} @ 60FPS<br>HDR: ENABLED | T-STAMP: 1044321
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # ── RIGHT COLUMN: Safety Gauge + Confidence Breakdown ──
        with col_info:
            # -- Safety Score Gauge --
            
            # Dynamically push the card down so it aligns with the image bypassing the Streamlit tab headers
            align_margin = "68px" if (safety_grid is not None and terrain_grid is not None) else "0px"
            
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:1.5rem 1rem; margin-top: {align_margin};">
                <div style="position:relative; width:160px; height:160px; margin:0 auto;">
                    <svg width="160" height="160">
                        <circle cx="80" cy="80" r="{radius}"
                                fill="none" stroke="rgba(124,58,237,0.15)" stroke-width="10"/>
                        <circle cx="80" cy="80" r="{radius}"
                                fill="none" stroke="{gauge_color}" stroke-width="10"
                                stroke-linecap="round"
                                stroke-dasharray="{circumference}"
                                stroke-dashoffset="{dash_offset}"
                                transform="rotate(-90 80 80)"/>
                    </svg>
                    <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); text-align:center;">
                        <div style="font-family:'Orbitron'; font-size:2.2rem; font-weight:800;">{safety_score}<span style="font-size:1rem; color:{gauge_color};">%</span></div>
                    </div>
                </div>
                <div style="margin-top:0.5rem; font-family:'Orbitron'; font-size:0.8rem; color:#C4B5FD; letter-spacing:1px; text-transform:uppercase;">Safety Score</div>
                <div style="margin-top:0.75rem; display:flex; align-items:center; justify-content:center; gap:6px;">
                    <span style="color:{gauge_color}; font-size:0.85rem;">✅</span>
                    <span style="color:{gauge_color}; font-size:0.75rem; font-weight:700; letter-spacing:1px;">{status_text}</span>
                </div>
                <p style="margin-top:0.5rem; font-size:0.85rem; color:#6B6B8A; line-height:1.4;">
                    {desc_text}
                </p>
            </div>
            """, unsafe_allow_html=True)

            # -- Confidence Breakdown Bars (from real model predictions) --
            if terrain_types:
                bars_html = ""
                for name, info in terrain_types.items():
                    bars_html += f'<div style="display:flex; align-items:center; margin-bottom:0.75rem;">'
                    bars_html += f'<span style="min-width:100px; font-size:0.85rem; color:#C0C0E0;">{name}</span>'
                    bars_html += f'<div style="flex:1; height:6px; background:rgba(124,58,237,0.1); border-radius:3px; margin:0 12px; overflow:hidden;">'
                    bars_html += f'<div style="width:{info["pct"]}%; height:100%; border-radius:3px; background:linear-gradient(90deg, {info["color"]}, {info["color"]}88);"></div>'
                    bars_html += f'</div>'
                    bars_html += f'<span style="font-family:Orbitron; font-size:0.8rem; color:{info["color"]}; min-width:40px; text-align:right;">{info["pct"]}%</span>'
                    bars_html += f'</div>'

                st.markdown(f"""<div class="glass-card" style="padding:1.25rem;">
                <div style="font-family:Orbitron; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; padding-bottom:0.75rem; border-bottom:1px solid rgba(124,58,237,0.2); margin-bottom:1rem;">Confidence Breakdown</div>
                {bars_html}
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown("""<div class="glass-card" style="padding:1.25rem; text-align:center;">
                <div style="font-family:Orbitron; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; padding-bottom:0.75rem; border-bottom:1px solid rgba(124,58,237,0.2); margin-bottom:1rem;">Confidence Breakdown</div>
                <p style="color:#6B6B8A; font-size:0.8rem;">Load a model to see predictions</p>
                </div>""", unsafe_allow_html=True)

        # ═══════════════════════════════════════
        #  ROW 2: Nested Dashboard Layout 
        # ═══════════════════════════════════════

        # Pre-calculations for dynamic data
        active_steps = max(1, int((safety_score / 100) * 8)) if safety_score > 0 else 0
        safe_color, warn_color_hex, danger_color = "#10B981", "#F59E0B", "#EF4444"
        
        slope_val = round(max(0, (100 - safety_score) * 0.35), 1)
        tract_val = min(99, safety_score + 15) if safety_score > 0 else 0
        slope_color = safe_color if slope_val < 15 else (warn_color_hex if slope_val < 25 else danger_color)
        t_color = safe_color if tract_val > 70 else (warn_color_hex if tract_val > 40 else danger_color)
        
        if terrain_types:
            sand_pct = terrain_types.get("Sand", {}).get("pct", 0)
            rock_pct = terrain_types.get("Rocks", {}).get("pct", 0)
            bedrock_pct = terrain_types.get("Bedrock", {}).get("pct", 0)
            
            if sand_pct > 25:
                warn_text = f"Loose regolith ({sand_pct}%). Slippage risk. Deviate."
                warn_icon, warn_color = "⚠", "#EF4444" 
            elif rock_pct > 25:
                warn_text = f"Dense rocks ({rock_pct}%). Abrasion risk. Deviate."
                warn_icon, warn_color = "⚠", "#EF4444"
            elif safety_score < 60:
                warn_text = "Mixed hazard terrain. Sub-optimal conditions."
                warn_icon, warn_color = "⚠", "#F59E0B"
            else:
                warn_text = "No critical surface hazards detected. Path clear."
                warn_icon, warn_color = "✅", "#10B981"
                
            obs_text = f"{bedrock_pct}% stable bedrock. Overall safety {safety_score}%. Environment nominal."
        else:
            warn_text, warn_icon, warn_color = "Awaiting model data...", "⏳", "#8B8B9B"
            obs_text = "Awaiting model data..."

        # 1. PATH PROJECTION ALPHA (Top Left Data Prep)
        waypoints_html = ""
        for i in range(1, 9):
            is_active = i <= active_steps
            cls = "active" if is_active else ""
            waypoints_html += f'<div class="waypoint {cls}">P{i}</div>'
            if i < 8:
                line_cls = "active" if i < active_steps else ""
                waypoints_html += f'<div class="waypoint-line {line_cls}"></div>'

        # Render everything natively in one unified block to bypass Streamlit column misalignment
        st.markdown(f"""
            <div style="display:grid; grid-template-columns: 2fr 1fr; gap:1.25rem; align-items:stretch; margin-top:0.5rem;">
            <div style="display:flex; flex-direction:column; gap:1.25rem;">
            <div class="glass-card path-projection" style="display:flex; flex-direction:column; justify-content:center;">
            <div style="display:flex; align-items:center; justify-content:space-between; margin-bottom:1.25rem;">
            <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-size:1.1rem;">✈️</span>
            <span style="font-family:'Orbitron'; font-size:0.85rem; font-weight:700;">Path Projection</span>
            </div>
            <span class="optimized-badge" style="background:rgba(16,185,129,0.15); color:#10B981;">ACTIVE</span>
            </div>
            <div class="waypoint-container">
            {waypoints_html}
            </div>
            </div>
            <div style="display:flex; gap:1.25rem; flex:1;">
            <div style="flex:1; background:rgba({int(warn_color[1:3], 16)},{int(warn_color[3:5], 16)},{int(warn_color[5:7], 16)},0.1); border:1px solid {warn_color}40; border-radius:12px; padding:1.25rem; display:flex; flex-direction:column; justify-content:center;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
            <span style="font-size:1.1rem; color:{warn_color};">{warn_icon}</span>
            <span style="font-family:'Orbitron'; font-size:0.9rem; font-weight:700; color:{warn_color}; letter-spacing:1px; text-transform:uppercase;">Warning</span>
            </div>
            <p style="color:#D1D5DB; font-size:0.85rem; line-height:1.5; margin:0;">
            {warn_text}
            </p>
            </div>
            <div style="flex:1; background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:12px; padding:1.25rem; display:flex; flex-direction:column; justify-content:center;">
            <div style="display:flex; align-items:center; gap:8px; margin-bottom:0.75rem;">
            <span style="font-size:1.1rem; color:#A78BFA;">🧠</span>
            <span style="font-family:'Orbitron'; font-size:0.9rem; font-weight:700; color:#E2E8F0; letter-spacing:1px; text-transform:uppercase;">AI Info</span>
            </div>
            <p style="color:#D1D5DB; font-size:0.85rem; line-height:1.5; margin:0;">
            {obs_text}
            </p>
            </div>
            </div>
            </div>
            <div class="glass-card" style="padding:1.25rem; display:flex; flex-direction:column; height:100%;">
            <div style="font-family:'Orbitron'; font-size:0.85rem; font-weight:700; text-transform:uppercase; letter-spacing:1.5px; padding-bottom:0.75rem; border-bottom:1px solid rgba(124,58,237,0.2); margin-bottom:1rem;">
            Stability
            </div>
            <div style="display:grid; grid-template-columns:1fr 1fr; gap:0.75rem; flex:1;">
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.75rem; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.7rem; color:#8B8B9B; margin-bottom:0.5rem; letter-spacing:1px;">SLOPE</div>
            <div style="font-size:1.2rem; font-weight:600; color:#E2E8F0;">{slope_val}°</div>
            <div style="font-size:0.65rem; color:{slope_color}; margin-top:0.25rem; font-weight:600;">{'Nominal' if slope_val < 15 else 'Caution'}</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.75rem; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.7rem; color:#8B8B9B; margin-bottom:0.5rem; letter-spacing:1px;">TRACT</div>
            <div style="font-size:1.2rem; font-weight:600; color:#E2E8F0;">{tract_val}%</div>
            <div style="font-size:0.65rem; color:{t_color}; margin-top:0.25rem; font-weight:600;">{'Optimal' if tract_val > 70 else 'Reduced'}</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.75rem; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.7rem; color:#8B8B9B; margin-bottom:0.5rem; letter-spacing:1px;">OCC</div>
            <div style="font-size:1.2rem; font-weight:600; color:#E2E8F0;">&lt; 5%</div>
            <div style="font-size:0.65rem; color:#10B981; margin-top:0.25rem; font-weight:600;">Clear</div>
            </div>
            <div style="background:rgba(255,255,255,0.03); border:1px solid rgba(255,255,255,0.05); border-radius:8px; padding:0.75rem; display:flex; flex-direction:column; justify-content:center;">
            <div style="font-size:0.7rem; color:#8B8B9B; margin-bottom:0.5rem; letter-spacing:1px;">PRES</div>
            <div style="font-size:1.2rem; font-weight:600; color:#E2E8F0;">6.1</div>
            <div style="font-size:0.65rem; color:#8B8B9B; margin-top:0.25rem; font-weight:600;">hPa</div>
            </div>
            </div>
            </div>
            </div>
            """, unsafe_allow_html=True)
        # ═══════════════════════════════════════
        #  RUN ANALYSIS BUTTON & PATHFINDING
        # ═══════════════════════════════════════
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("RUN FULL TERRAIN ANALYSIS", use_container_width=True, type="primary")

        if run_btn:
            if 'model' not in st.session_state or st.session_state['model'] is None:
                st.error("Please navigate to the sidebar and load a .keras model first.")
            else:
                with st.spinner("🤖 Connecting to AresNav Core... Routing A* Path..."):
                    # 1. Imports for pathfinding
                    from utils.pathfinding import astar_pathfinding, smooth_path
                    from utils.plotting import plot_smoothed_path_on_image
                    from utils.ml_engine import NAVIGABILITY_MAP, CLASS_NAMES
                    
                    # 2. Run initial ML Prediction (now grabbing the grids too!)
                    model = st.session_state['model']
                    safety_score, terrain_report, terrain_grid, safety_grid = run_terrain_prediction(img_rgb, model)
                    
                    # 3. Build Cost Map from Navigability
                    nav_grid = np.zeros_like(terrain_grid, dtype=float)
                    for r in range(terrain_grid.shape[0]):
                        for c in range(terrain_grid.shape[1]):
                            terrain_id = terrain_grid[r, c]
                            nav_grid[r, c] = NAVIGABILITY_MAP[terrain_id]
                            
                    # Explicit, hand-tuned Cost Profile to absolutely maximize time spent in green zones.
                    cost_grid = np.zeros_like(terrain_grid, dtype=float)
                    for r in range(terrain_grid.shape[0]):
                        for c in range(terrain_grid.shape[1]):
                            tid = terrain_grid[r, c]
                            if tid == 0: cost_grid[r, c] = 1.0       # Bedrock (Light Green) - Ideal
                            elif tid == 3: cost_grid[r, c] = 3.0     # Pebbles (Dark Green) - Okay
                            elif tid == 1: cost_grid[r, c] = 200.0   # Rocks (Red/Brown) - Bad
                            else: cost_grid[r, c] = 800.0            # Sand (Mud) - Avoid at all costs

                    # 4. Global Optimal Path Discovery
                    # To prevent the rover from starting in a "safe trap" surrounded by danger,
                    # we calculate every possible path from bottom to top and pick the global best.
                    bottom_row = terrain_grid.shape[0] - 1
                    best_path = None
                    best_cost = float('inf')

                    # The absolute top row (row 0) is usually the Martian sky or distant mountains!
                    # Pathing a rover into the sky looks absurd. We should set the goal row slightly lower (midground).
                    goal_row = max(1, terrain_grid.shape[0] // 3)

                    for start_col in range(terrain_grid.shape[1]):
                        for goal_col in range(terrain_grid.shape[1]):
                            start_node = (bottom_row, start_col)
                            goal_node = (goal_row, goal_col)
                            
                            path = astar_pathfinding(cost_grid, start_node, goal_node)
                            if path is not None:
                                path_cost = sum([cost_grid[r, c] for r, c in path])
                                if path_cost < best_cost:
                                    best_cost = path_cost
                                    best_path = path

                    raw_path = best_path

                    if raw_path is None:
                        st.error("🚨 CRITICAL: No safe path exists across this terrain. Reroute required.")
                    else:
                        st.success(f"✅ Optimal Global Path calculated! Lowest hazard score.")
                        
                        # Smooth the path for a realistic rover turning radius
                        path_img_coords, smooth_img_coords = smooth_path(raw_path, safety_grid, img_rgb.shape)
                        
                        # Render the final futuristic map
                        st.markdown("<br><hr style='border:1px solid #2B2B3B; margin-bottom: 2rem;'>", unsafe_allow_html=True)
                        st.markdown("""
                        <div class="glass-card" style="margin-bottom: 2rem; padding: 1.5rem;">
                            <h3 style="margin-top:0; color:white;">🗺️ A* Pathfinding Projection & Core Navigation Module</h3>
                            <p style="color: #A0A0C0; font-size: 0.95rem; margin-bottom:0; line-height: 1.6;">
                                This module calculates the safest optimal route across the Martian surface using the A* pathfinding algorithm. The system balances <b>Terrain Safety</b> against <b>Traversal Distance</b>.
                                Explore the tabs below to view the Navigability cost maps, the Smoothed trajectory projection, and the active Look-Ahead decision nodes the rover calculates in real-time.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        from utils.plotting import plot_path_on_grid, plot_decision_visualization, plot_decision_on_grid
                        
                        # Pick a point mid-way along the path to simulate a look-ahead
                        mid_index = max(0, len(raw_path) // 4)
                        
                        # Organize the 5 output visualizations into Streamlit Tabs exactly like the top section
                        ptab1, ptab2, ptab3, ptab4, ptab5 = st.tabs([
                            "Navigability Graph", 
                            "Safety Grid", 
                            "Decision Grid", 
                            "Smoothed Path", 
                            "Decision Overlay"
                        ])
                        
                        with ptab1:
                            fig1 = plot_path_on_grid(nav_grid, raw_path, cmap="YlGn", title="")
                            st.pyplot(fig1)
                            
                        with ptab2:
                            fig2 = plot_path_on_grid(safety_grid, raw_path, cmap="RdYlGn", title="")
                            st.pyplot(fig2)
                            
                        with ptab3:
                            fig5 = plot_decision_on_grid(safety_grid, nav_grid, raw_path, mid_index)
                            st.pyplot(fig5)
                            
                        with ptab4:
                            final_fig = plot_smoothed_path_on_image(img_rgb, safety_grid, nav_grid, path_img_coords, smooth_img_coords)
                            st.pyplot(final_fig)
                            
                        with ptab5:
                            fig4 = plot_decision_visualization(img_rgb, safety_grid, nav_grid, raw_path, mid_index)
                            st.pyplot(fig4)
                            
                        # 4. Deep Learning Explainability (Grad-CAM)
                        st.markdown("<br><hr style='border:1px solid #2B2B3B; margin-bottom: 2rem;'>", unsafe_allow_html=True)
                        
                        st.markdown("""
                        <div class="glass-card" style="margin-bottom: 2rem; padding: 1.5rem;">
                            <h3 style="margin-top:0; color:white;">🧠 Neural Network Attention (Grad-CAM)</h3>
                            <p style="color: #A0A0C0; font-size: 0.95rem; margin-bottom:0; line-height: 1.6;">
                                This module visualizes exactly <b>where</b> the AI is looking to make its terrain predictions. 
                                The heatmap colors represent neural activation: <span style="color:#EF4444; font-weight:bold;">Red/Orange zones</span> indicate areas of high focus that heavily influenced the model, while <span style="color:#3B82F6; font-weight:bold;">Blue zones</span> are ignored background. This provides crucial transparency into the rover's local decision-making!
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        from utils.ml_engine import generate_gradcam, PATCH_SIZE, CLASS_NAMES
                        from utils.plotting import plot_gradcam, BG_COLOR
                        import matplotlib.pyplot as plt
                        
                        # Grab a prominent patch (center of image)
                        h, w = img_rgb.shape[:2]
                        cy, cx = h // 2, w // 2
                        half_p = PATCH_SIZE // 2
                        
                        if h < PATCH_SIZE or w < PATCH_SIZE:
                            patch_rgb = cv2.resize(img_rgb, (PATCH_SIZE, PATCH_SIZE))
                        else:
                            patch_rgb = img_rgb[max(0, cy-half_p):cy+half_p, max(0, cx-half_p):cx+half_p]
                            
                        if patch_rgb.shape[:2] != (PATCH_SIZE, PATCH_SIZE):
                            patch_rgb = cv2.resize(patch_rgb, (PATCH_SIZE, PATCH_SIZE))
                            
                        with st.spinner("🧠 Generating Explanatory Maps..."):
                            heatmap, pred_idx, err = generate_gradcam(model, patch_rgb)
                            
                        if err:
                            st.warning(f"Grad-CAM unavailable: {err}")
                        elif heatmap is not None:
                            c_name = CLASS_NAMES[pred_idx]
                            
                            col_cam1, col_cam2 = st.columns([1, 2.5])
                            with col_cam1:
                                fig_top, ax_top = plt.subplots(figsize=(4, 4))
                                ax_top.imshow(heatmap, cmap="jet")
                                ax_top.set_title(f"Focus Heatmap", color='white', fontsize=13, pad=10)
                                ax_top.axis("off")
                                fig_top.patch.set_facecolor(BG_COLOR)
                                st.pyplot(fig_top)
                                
                            with col_cam2:
                                fig_bot = plot_gradcam(patch_rgb, heatmap, c_name)
                                st.pyplot(fig_bot)

