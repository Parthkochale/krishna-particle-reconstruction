"""
╔══════════════════════════════════════════════════════════════════╗
║        KRISHNA PARTICLE RECONSTRUCTION — M9 GOD TIER            ║
║   Cinematic neon particle animation → photorealistic Krishna     ║
╚══════════════════════════════════════════════════════════════════╝

Visual targets:
  • Mid-formation: neon chaos with Krishna silhouette glowing through
  • Final: deep cobalt Krishna, neon magenta hair, teal arms,
    surrounded by a soft particle halo — indistinguishable from art

Controls:
  ESC / Q  — Exit
  SPACE    — Restart
"""

import cv2
import numpy as np
import os, sys, time

# ═══════════════════════════════════════════════════════════════
#  CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CANVAS_W, CANVAS_H   = 900, 900
TARGET_W, TARGET_H   = 540, 680        # Max Krishna display size
NUM_PARTICLES        = 20000
FPS_TARGET           = 60
WINDOW_NAME          = "✦ Little Krishna — Particle Reconstruction ✦"

# Phase durations (seconds)
PH1 = 2.2    # Scattered chaos
PH2 = 3.0    # Swirling storm
PH3 = 4.5    # Attraction begins
PH4 = 3.5    # Image forms
PH5 = 2.0    # Final snap
PH6 = 9999.  # Alive forever

T1 = 0.0
T2 = T1 + PH1
T3 = T2 + PH2
T4 = T3 + PH3
T5 = T4 + PH4
T6 = T5 + PH5

# Neon palette for chaos phase (BGR)
NEON = np.array([
    [255,  40, 200],  # hot pink
    [ 40, 220, 255],  # cyan
    [200,  40, 255],  # purple
    [255, 180,  20],  # gold
    [ 20, 255, 120],  # lime
    [255,  80,  20],  # orange
    [ 80, 160, 255],  # sky
    [255,  20, 100],  # crimson
    [180, 255,  40],  # yellow-green
    [ 40, 100, 255],  # blue
], dtype=np.float32)

# ═══════════════════════════════════════════════════════════════
#  IMAGE LOADING
# ═══════════════════════════════════════════════════════════════

def find_asset():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for rel in [
        "../assets/krishna.png",
        "assets/krishna.png",
        "../../assets/krishna.png",
    ]:
        p = os.path.normpath(os.path.join(script_dir, rel))
        if os.path.exists(p):
            return p
    # also check cwd
    cwd_p = os.path.join(os.getcwd(), "assets", "krishna.png")
    if os.path.exists(cwd_p):
        return cwd_p
    print("ERROR: assets/krishna.png not found")
    sys.exit(1)


def load_and_process():
    path = find_asset()
    print(f"Loading: {path}")
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        print("ERROR: cv2 could not read image"); sys.exit(1)

    # --- extract BGR + alpha ---
    if raw.ndim == 2:
        bgr   = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)
        alpha = np.ones(bgr.shape[:2], np.float32)
    elif raw.shape[2] == 4:
        bgr   = raw[:, :, :3].astype(np.float32)
        alpha = raw[:, :, 3].astype(np.float32) / 255.0
    else:
        bgr   = raw.astype(np.float32)
        alpha = np.ones(bgr.shape[:2], np.float32)

    # --- if alpha is mostly opaque, detect background via color ---
    if alpha.mean() > 0.92:
        hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
        sat = hsv[:, :, 1] / 255.0
        val = hsv[:, :, 2] / 255.0
        bg  = (sat < 0.10) & (val > 0.90)          # near-white / checker
        alpha = 1.0 - bg.astype(np.float32)
        alpha = cv2.GaussianBlur(alpha, (7, 7), 2.5)
        alpha = np.clip(alpha, 0, 1)

    # --- resize preserving aspect ---
    h0, w0 = bgr.shape[:2]
    sc     = min(TARGET_W / w0, TARGET_H / h0)
    nw, nh = int(w0 * sc), int(h0 * sc)
    bgr    = cv2.resize(bgr,   (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    alpha  = cv2.resize(alpha, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    bgr_f  = bgr / 255.0                           # float 0-1

    # Canvas offsets (centred)
    ox = (CANVAS_W - nw) // 2
    oy = (CANVAS_H - nh) // 2

    # --- build FULL display image (for final blend) ---
    # Boost saturation for neon look
    img_hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    img_hsv[:, :, 1] = np.clip(img_hsv[:, :, 1] * 1.9, 0, 255)
    img_hsv[:, :, 2] = np.clip(img_hsv[:, :, 2] * 1.1, 0, 255)
    bgr_neon = cv2.cvtColor(img_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # Full-canvas target image (with alpha composited on black)
    target_canvas = np.zeros((CANVAS_H, CANVAS_W, 3), np.float32)
    for c in range(3):
        target_canvas[oy:oy+nh, ox:ox+nw, c] = bgr_neon[:, :, c] * alpha

    # --- sample particle target positions ---
    THRESH = 0.30
    ys, xs = np.where(alpha > THRESH)
    colors  = bgr_neon[ys, xs]                     # [N_px, 3] BGR float
    cxs     = (xs + ox).astype(np.float32)
    cys     = (ys + oy).astype(np.float32)
    positions = np.stack([cxs, cys], axis=1)

    print(f"Subject pixels: {len(ys):,}  |  display size: {nw}×{nh}")
    return positions, colors, target_canvas, ox, oy, nw, nh


# ═══════════════════════════════════════════════════════════════
#  PARTICLE SYSTEM
# ═══════════════════════════════════════════════════════════════

class Particles:
    def __init__(self, positions, colors):
        N = NUM_PARTICLES
        self.N = N
        n_px = len(positions)

        # Assign targets
        idx = np.random.choice(n_px, N, replace=(N > n_px))
        self.target    = positions[idx].astype(np.float32)
        self.img_color = colors[idx].astype(np.float32)

        # Chaos neon color per particle
        pidx = np.random.randint(0, len(NEON), N)
        self.neon_color = (NEON[pidx] / 255.0).astype(np.float32)

        # Starting color = neon
        self.color = self.neon_color.copy()

        # Initial positions: scattered across canvas + exploding from edges
        n_rand  = int(N * 0.80)
        n_edge  = N - n_rand

        pos_r          = np.random.rand(n_rand, 2).astype(np.float32)
        pos_r[:, 0]   *= CANVAS_W
        pos_r[:, 1]   *= CANVAS_H

        # Edge bursts
        side  = np.random.randint(0, 4, n_edge)
        ex    = np.where(side==0, np.random.rand(n_edge)*CANVAS_W,
                np.where(side==1, np.random.rand(n_edge)*CANVAS_W,
                np.where(side==2, np.full(n_edge, -50.0),
                                  np.full(n_edge, CANVAS_W+50.0)))).astype(np.float32)
        ey    = np.where(side==0, np.full(n_edge, -50.0),
                np.where(side==1, np.full(n_edge, CANVAS_H+50.0),
                         np.random.rand(n_edge)*CANVAS_H)).astype(np.float32)
        pos_e = np.stack([ex, ey], axis=1)

        self.pos = np.concatenate([pos_r, pos_e], axis=0).astype(np.float32)

        # Initial velocity: random explosion
        ang         = np.random.rand(N) * 2 * np.pi
        spd         = np.random.uniform(1.0, 6.0, N).astype(np.float32)
        self.vel    = np.stack([np.cos(ang)*spd, np.sin(ang)*spd], axis=1).astype(np.float32)

        # Per-particle uniqueness
        self.phase_off  = np.random.rand(N).astype(np.float32) * 80.0
        self.radius     = np.random.choice([1,1,1,2,2,3], N).astype(np.int32)

        # Swirl centre
        self.scx = float(CANVAS_W) / 2
        self.scy = float(CANVAS_H) / 2

        # brightness flicker per particle
        self.flicker_f  = np.random.uniform(2.0, 8.0, N).astype(np.float32)

    def update(self, t, dt):
        att, cblend, swirl, damp = self._params(t)

        # --- Turbulence (cheap sinusoidal noise) ---
        ang_n = (np.sin(self.pos[:, 0] * 0.011 + t * 1.2 + self.phase_off) +
                 np.cos(self.pos[:, 1] * 0.014 + t * 0.8 + self.phase_off * 1.4)) * np.pi
        nx = np.cos(ang_n) * 1.4
        ny = np.sin(ang_n) * 1.4

        # --- Swirl force ---
        dx_s  = self.pos[:, 0] - self.scx
        dy_s  = self.pos[:, 1] - self.scy
        d_s   = np.hypot(dx_s, dy_s) + 1e-6
        # orbital (perpendicular)
        svx   = (-dy_s / d_s) * swirl
        svy   = ( dx_s / d_s) * swirl
        # gentle inward pull so particles stay visible
        svx  -= (dx_s / d_s) * swirl * 0.06
        svy  -= (dy_s / d_s) * swirl * 0.06

        # --- Attraction force ---
        dx_t  = self.target[:, 0] - self.pos[:, 0]
        dy_t  = self.target[:, 1] - self.pos[:, 1]
        d_t   = np.hypot(dx_t, dy_t) + 1e-6
        amag  = np.clip(d_t * att, 0, 14.0)
        avx   = (dx_t / d_t) * amag
        avy   = (dy_t / d_t) * amag

        # --- Living wiggle (phase 6 only) ---
        wvx = wvy = 0.0
        if t > T6:
            wvx = np.sin(t * 3.3 + self.phase_off * 2.1) * 0.4
            wvy = np.cos(t * 2.8 + self.phase_off * 1.8) * 0.4

        # Integrate velocity
        scale = dt * 60.0
        self.vel[:, 0] += (svx + avx + nx + wvx) * scale
        self.vel[:, 1] += (svy + avy + ny + wvy) * scale
        self.vel       *= damp

        # Speed cap
        spd    = np.hypot(self.vel[:, 0], self.vel[:, 1]) + 1e-9
        cap    = 20.0
        mask_c = spd > cap
        self.vel[mask_c] = self.vel[mask_c] / spd[mask_c, None] * cap

        self.pos += self.vel * scale

        # Boundary soft clamp
        mg = 100.0
        self.pos[:, 0] = np.clip(self.pos[:, 0], -mg, CANVAS_W + mg)
        self.pos[:, 1] = np.clip(self.pos[:, 1], -mg, CANVAS_H + mg)

        # Hard snap once very close (phase 5+)
        if t > T5:
            close = d_t < 2.8
            self.pos[close] = self.target[close]
            self.vel[close] = 0.0

        # --- Color blend chaos→image ---
        cb           = float(np.clip(cblend, 0, 1))
        self.color   = self.neon_color * (1.0 - cb) + self.img_color * cb

        # Flicker during chaos / storm
        if t < T3:
            amp     = 0.18 * max(0, 1.0 - (t / T3))
            flick   = 1.0 + amp * np.sin(t * self.flicker_f + self.phase_off)[:, None]
            self.color = np.clip(self.color * flick, 0, 1)

    def _params(self, t):
        # returns (attraction, color_blend, swirl, damping)
        if t < T2:
            return 0.0, 0.0, 0.0, 0.87
        elif t < T3:
            p = (t - T2) / PH2
            s = _ss(p)
            return 0.0, 0.0, 3.5 * s, 0.84
        elif t < T4:
            p  = (t - T3) / PH3
            s  = _ss(p)
            return 0.038 * s * 2.2, s * 0.35, 3.5 * (1 - s * 0.75), 0.82
        elif t < T5:
            p  = (t - T4) / PH4
            s  = _ss(p)
            return 0.038 * (2.2 + s * 5.0), 0.35 + s * 0.55, 3.5 * 0.25 * (1-s), 0.80
        elif t < T6:
            p  = (t - T5) / PH5
            s  = _ss(p)
            return 0.038 * 14.0, 0.90 + s * 0.10, 0.0, 0.76
        else:
            return 0.038 * 6.0, 1.0, 0.0, 0.78


def _ss(x):
    x = np.clip(x, 0, 1)
    return x * x * (3 - 2 * x)


# ═══════════════════════════════════════════════════════════════
#  RENDERING — THE KEY TO GOD-TIER OUTPUT
# ═══════════════════════════════════════════════════════════════

def render(particles, canvas, glow_buf, target_canvas, t):
    """
    Multi-layer render:
      Layer 1 — Particle dots (neon chaos)
      Layer 2 — Glow bloom (Gaussian additive)
      Layer 3 — Target image reveal (alpha-blended on top of particles)
                so final looks like real neon-lit artwork
    """
    canvas[:] = 0

    N   = particles.N
    px  = particles.pos[:, 0].astype(np.int32)
    py  = particles.pos[:, 1].astype(np.int32)
    ok  = (px >= 0) & (px < CANVAS_W) & (py >= 0) & (py < CANVAS_H)
    pxv = px[ok]; pyv = py[ok]
    cv  = particles.color[ok]
    rv  = particles.radius[ok]

    # ── Draw particles ────────────────────────────────────────────
    # radius 1: vectorised pixel write
    m1 = rv == 1
    if m1.any():
        c8 = (cv[m1] * 255).astype(np.uint8)
        canvas[pyv[m1], pxv[m1]] = c8

    # radius 2 & 3: circles
    for r in (2, 3):
        mr = rv == r
        if not mr.any(): continue
        xr = pxv[mr]; yr = pyv[mr]
        c8 = (cv[mr] * 255).astype(np.uint8)
        for i in range(len(xr)):
            col = (int(c8[i,0]), int(c8[i,1]), int(c8[i,2]))
            cv2.circle(canvas, (xr[i], yr[i]), r, col, -1, cv2.LINE_AA)

    # ── Glow bloom ────────────────────────────────────────────────
    # Two-pass: tight sharp glow + wide soft halo
    blur1 = cv2.GaussianBlur(canvas, (5,  5),  1.5)
    blur2 = cv2.GaussianBlur(canvas, (15, 15), 6.0)
    bloom = (blur1.astype(np.float32) * 0.55 +
             blur2.astype(np.float32) * 0.30)
    np.add(canvas, bloom.astype(np.uint8), out=canvas, casting="unsafe")
    np.clip(canvas, 0, 255, out=canvas)

    # ── Image reveal blend ────────────────────────────────────────
    # As particles settle, fade in the ACTUAL image on top.
    # This is what makes the final look like real neon art, not dots.
    #
    # img_alpha: how opaque the real image is
    #   - 0  during chaos (pure particles)
    #   - 0  until phase 4 starts
    #   - rises from 0→1 across phase 4+5
    #   - hits 1.0 by end of phase 5

    img_alpha = 0.0
    if t >= T4:
        if t < T5:
            img_alpha = _ss((t - T4) / PH4) * 0.85
        elif t < T6:
            img_alpha = 0.85 + _ss((t - T5) / PH5) * 0.15
        else:
            img_alpha = 1.0

    if img_alpha > 0.001:
        # target_canvas is float32 BGR 0-1
        tc8 = (target_canvas * 255).astype(np.uint8)
        # Additive blend: particles visible under image during transition,
        # image takes over fully by the end
        cv2.addWeighted(canvas, 1.0 - img_alpha * 0.7,
                        tc8,   img_alpha,
                        0, canvas)
        # Extra glow on the image reveal
        if img_alpha > 0.3:
            glow_img = cv2.GaussianBlur(tc8, (9, 9), 3.5)
            blend_g  = (img_alpha - 0.3) / 0.7 * 0.35
            cv2.addWeighted(canvas, 1.0, glow_img, blend_g, 0, canvas)

    # ── HUD phase label ───────────────────────────────────────────
    lbl = _label(t)
    if lbl:
        cv2.putText(canvas, lbl, (16, CANVAS_H - 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (70,70,70), 1, cv2.LINE_AA)


def _label(t):
    if   t < T2: return "PHASE I  ·  CHAOS"
    elif t < T3: return "PHASE II  ·  STORM"
    elif t < T4: return "PHASE III  ·  ATTRACTION"
    elif t < T5: return "PHASE IV  ·  FORMATION"
    elif t < T6: return "PHASE V  ·  DEFINITION"
    else:         return "\u2736  JAY SHRI KRISHNA  \u2736"


# ═══════════════════════════════════════════════════════════════
#  MAIN LOOP
# ═══════════════════════════════════════════════════════════════

def main():
    positions, colors, target_canvas, ox, oy, nw, nh = load_and_process()

    glow_buf = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)
    canvas   = np.zeros((CANVAS_H, CANVAS_W, 3), np.uint8)

    def restart():
        np.random.seed()
        return Particles(positions, colors)

    particles = restart()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, CANVAS_W, CANVAS_H)

    t_start = time.perf_counter()
    t_last  = t_start
    frame_t = 1.0 / FPS_TARGET

    print("\n╔════════════════════════════════╗")
    print("║  ESC / Q  — Exit               ║")
    print("║  SPACE    — Restart            ║")
    print("╚════════════════════════════════╝\n")

    while True:
        now = time.perf_counter()
        t   = now - t_start
        dt  = min(now - t_last, 0.05)
        t_last = now

        # Drift swirl centre for drama
        particles.scx = CANVAS_W/2 + np.sin(t * 0.28) * 55
        particles.scy = CANVAS_H/2 + np.cos(t * 0.22) * 38

        particles.update(t, dt)
        render(particles, canvas, glow_buf, target_canvas, t)

        cv2.imshow(WINDOW_NAME, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break
        if key == ord(' '):
            particles = restart()
            t_start   = time.perf_counter()
            t_last    = t_start

        elapsed = time.perf_counter() - now
        wait    = frame_t - elapsed
        if wait > 0:
            time.sleep(wait)

    cv2.destroyAllWindows()
    print("✦ Jay Shri Krishna ✦")


if __name__ == "__main__":
    main()