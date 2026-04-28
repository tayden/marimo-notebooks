import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from scipy.optimize import least_squares
    from scipy.linalg import lstsq

    return least_squares, lstsq, mo, np, plt


@app.cell
def _():
    # Shared image dimensions used across all sections
    W, H = 640, 360
    return H, W


@app.cell
def _(mo):
    mo.md(r"""
    # Icefield Projector: How Calibration Works

    This notebook explains the calibration system that allows the icefield projector to align
    a projected image with a 3D-printed terrain model. All examples use synthetic dummy data —
    no real data files needed.

    **Sections**
    1. The Pinhole Camera Model — intrinsics, throw ratio, lens shift
    2. The Projector Pose — position, look direction, view matrix
    3. Lens Distortion — radial distortion and the k1/k2 coefficients
    4. The 3D Surface — georeferencing, the DEM, and z-exaggeration
    5. Ground Control Points and Pose Optimisation — reprojection error, Levenberg-Marquardt
    6. Polynomial Residual Correction — absorbing the remaining error with poly2
    7. Image Draping and the Inverse Warp — Delaunay triangulation, simplex filtering, cv2.remap
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. The Pinhole Camera Model

    A projector is a **pinhole camera running in reverse**: instead of recording light from the
    world onto a sensor, it casts light from an image plane into the world. The maths is identical
    in both cases.

    ### Intrinsic parameters

    The *intrinsic* parameters describe the optics alone — independent of where the projector is
    pointing. Two parameters dominate:

    **Throw ratio** is the ratio of throw distance (projector-to-screen distance) to image width.
    A short throw ratio (< 0.5) gives a very wide beam — great for tight spaces. A long throw
    ratio (> 1.0) gives a narrow beam, like a standard lecture projector. From it we get the
    horizontal field of view and the focal length in pixels:

    $$\text{hfov} = 2\arctan\!\left(\tfrac{1}{2 \cdot \text{throw\_ratio}}\right)$$

    $$f_x = \frac{W}{2\tan(\text{hfov}/2)} = W \cdot \text{throw\_ratio}$$

    **Lens shift** moves the principal point (the pixel the lens axis passes through) away from the
    image centre *without* tilting the projector body. Positive vertical shift moves the image
    upward relative to the lens:

    $$c_x = \tfrac{W}{2} - \text{lens\_shift\_h} \cdot W \qquad
      c_y = \tfrac{H}{2} - \text{lens\_shift\_v} \cdot H$$

    A 3D point at camera-space position $(X_c,\,Y_c,\,Z_c)$ then maps to pixel $(u, v)$:

    $$u = f_x \cdot \frac{X_c}{Z_c} + c_x \qquad v = f_y \cdot \frac{Y_c}{Z_c} + c_y$$

    > **In the code:** `configs.py` → `ProjectorConfig.compute_intrinsic_matrix()` (lines 61–72)

    Use the sliders below to feel what each parameter does to the projected image boundary.
    """)
    return


@app.cell
def _(mo):
    throw_ratio_s = mo.ui.slider(0.30, 1.50, step=0.05, value=0.58, label="Throw ratio")
    lens_shift_v_s = mo.ui.slider(-0.40, 0.40, step=0.05, value=0.00, label="Lens shift — vertical")
    lens_shift_h_s = mo.ui.slider(-0.40, 0.40, step=0.05, value=0.00, label="Lens shift — horizontal")
    return lens_shift_h_s, lens_shift_v_s, throw_ratio_s


@app.cell
def _(H, W, lens_shift_h_s, lens_shift_v_s, mo, np, plt, throw_ratio_s):
    tr = throw_ratio_s.value
    lsv = lens_shift_v_s.value
    lsh = lens_shift_h_s.value

    fx1 = W * tr
    cx1 = W / 2 - lsh * W
    cy1 = H / 2 - lsv * H
    hfov_deg1 = np.degrees(2 * np.arctan(1 / (2 * tr)))

    corners1 = np.array([
        [-500.0, -500.0 * H / W, 1000.0],
        [ 500.0, -500.0 * H / W, 1000.0],
        [ 500.0,  500.0 * H / W, 1000.0],
        [-500.0,  500.0 * H / W, 1000.0],
        [-500.0, -500.0 * H / W, 1000.0],
    ])
    u1 = fx1 * corners1[:, 0] / corners1[:, 2] + cx1
    v1 = fx1 * corners1[:, 1] / corners1[:, 2] + cy1

    fig1, ax1 = plt.subplots(figsize=(7, 4.5))
    fig1.patch.set_facecolor("#1a1a2e")
    ax1.set_facecolor("#111")
    ax1.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=1.5))
    ax1.plot(u1, v1, "w-", lw=2, label="Projected scene boundary")
    ax1.plot(cx1, cy1, "r+", ms=14, mew=2.5, label=f"Principal point ({cx1:.0f}, {cy1:.0f})")
    ax1.axvline(W / 2, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax1.axhline(H / 2, color="gray", ls="--", lw=0.8, alpha=0.4)
    ax1.set_xlim(-20, W + 20)
    ax1.set_ylim(H + 20, -20)
    ax1.set_xlabel("Pixel x", color="white")
    ax1.set_ylabel("Pixel y", color="white")
    ax1.tick_params(colors="white")
    ax1.set_title(
        f"throw_ratio={tr:.2f}  →  hfov={hfov_deg1:.1f}°  |  fx={fx1:.0f} px",
        color="white", fontsize=10,
    )
    ax1.legend(facecolor="#333", labelcolor="white", fontsize=9)
    plt.tight_layout()

    info1 = mo.callout(
        mo.md(
            f"**fx = {fx1:.0f} px** — a higher throw ratio means a longer focal length (narrower beam).  "
            f"The principal point sits at ({cx1:.0f}, {cy1:.0f}); lens shift moves it without tilting the projector."
        ),
        kind="info",
    )
    return fig1, info1, tr


@app.cell
def _(fig1, info1, lens_shift_h_s, lens_shift_v_s, mo, throw_ratio_s):
    mo.vstack([
        mo.vstack([throw_ratio_s, lens_shift_v_s, lens_shift_h_s]),
        fig1,
        info1,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The Projector Pose (Extrinsics)

    The *extrinsic* parameters describe *where* the projector is in the world and *which way* it
    points. Three vectors define the pose:

    - **`position`** — 3D location of the projector in model space (millimetres)
    - **`look_direction`** — unit vector pointing from the projector toward the scene
    - **`up_vector`** — defines which edge of the image is "up"

    These form the **view matrix** — a rigid-body transform that converts world coordinates into
    camera-local coordinates. In camera space the projector sits at the origin looking along +Z.

    The construction is:

    ```
    forward  = normalise(look_direction)
    right    = normalise(forward × up_vector)
    up_true  = right × forward          # re-orthogonalised
    R = [right, −up_true, forward]      # 3×3 rotation matrix
    t = −R · position                   # translation column
    ```

    A world point **p** transforms to camera space as **p**_c = R**p** + t, then projects through
    the intrinsic matrix.

    > **In the code:** `configs.py` → `ProjectorConfig.compute_view_matrix()` (lines 47–59)
    > `geometric.py` → `IcefieldProjectionMapper.project_points()` (lines 320–358)

    Adjust the sliders to move and tilt the camera and see how the projected terrain shifts.
    """)
    return


@app.cell
def _(mo):
    cam_dx_s = mo.ui.slider(-80, 80, step=5, value=0, label="Camera X offset (mm)")
    cam_dy_s = mo.ui.slider(-80, 80, step=5, value=0, label="Camera Y offset (mm)")
    cam_dz_s = mo.ui.slider(-150, 150, step=10, value=0, label="Camera height offset (mm)")
    cam_yaw_s = mo.ui.slider(-25, 25, step=1, value=0, label="Yaw (°)")
    cam_pitch_s = mo.ui.slider(-20, 20, step=1, value=0, label="Pitch (°)")
    return cam_dx_s, cam_dy_s, cam_dz_s, cam_pitch_s, cam_yaw_s


@app.cell
def _(H, W, cam_dx_s, cam_dy_s, cam_dz_s, cam_pitch_s, cam_yaw_s, np, plt, tr):
    gx2 = np.linspace(0, 200, 10)
    gy2 = np.linspace(0, 200, 10)
    GX2, GY2 = np.meshgrid(gx2, gy2)
    GZ2 = 8 * np.sin(GX2 / 40) * np.cos(GY2 / 40) + 5
    pts2 = np.stack([GX2.ravel(), GY2.ravel(), GZ2.ravel()], axis=1)

    base_pos2 = np.array([100.0, -20.0, 380.0])
    pos2 = base_pos2 + np.array([cam_dx_s.value, cam_dy_s.value, cam_dz_s.value], dtype=float)

    yaw_r2 = np.radians(cam_yaw_s.value)
    pitch_r2 = np.radians(cam_pitch_s.value)

    # Base look direction from the nominal position — position offsets are pure translations
    target2 = np.array([100.0, 100.0, 0.0])
    fwd2 = target2 - base_pos2
    fwd2 /= np.linalg.norm(fwd2)
    up_v2 = np.array([0.0, 1.0, 0.0])

    Ryaw2 = np.array([
        [np.cos(yaw_r2), -np.sin(yaw_r2), 0],
        [np.sin(yaw_r2),  np.cos(yaw_r2), 0],
        [0, 0, 1],
    ])
    fwd2 = Ryaw2 @ fwd2

    right2 = np.cross(fwd2, up_v2); right2 /= np.linalg.norm(right2)
    c_p2, s_p2 = np.cos(pitch_r2), np.sin(pitch_r2)
    Rpitch2 = (c_p2 * np.eye(3)
               + s_p2 * np.cross(right2, np.eye(3))
               + (1 - c_p2) * np.outer(right2, right2))
    fwd2 = Rpitch2 @ fwd2 / np.linalg.norm(Rpitch2 @ fwd2)

    right2 = np.cross(fwd2, up_v2); right2 /= np.linalg.norm(right2)
    up_t2 = np.cross(right2, fwd2)
    R2 = np.array([right2, -up_t2, fwd2])
    t_vec2 = -R2 @ pos2

    pts_cam2 = (R2 @ pts2.T).T + t_vec2
    fx2 = W * tr
    valid2 = pts_cam2[:, 2] > 0
    u2 = np.where(valid2, fx2 * pts_cam2[:, 0] / pts_cam2[:, 2] + W / 2, np.nan)
    v2 = np.where(valid2, fx2 * pts_cam2[:, 1] / pts_cam2[:, 2] + H / 2, np.nan)

    fdepth2 = 150.0
    frust_norm2 = np.array([
        [-1 / tr, -1 / tr * H / W, 1],
        [ 1 / tr, -1 / tr * H / W, 1],
        [ 1 / tr,  1 / tr * H / W, 1],
        [-1 / tr,  1 / tr * H / W, 1],
    ])
    frust_world2 = (R2.T @ (fdepth2 * frust_norm2.T - t_vec2[:, None])).T

    fig2 = plt.figure(figsize=(11, 4.5))
    fig2.patch.set_facecolor("#1a1a2e")

    ax2_3d = fig2.add_subplot(1, 2, 1, projection="3d")
    ax2_3d.scatter(GX2.ravel(), GY2.ravel(), GZ2.ravel(),
                   c=GZ2.ravel(), cmap="terrain", s=12, alpha=0.7)
    ax2_3d.scatter(*pos2, color="red", s=60, zorder=5, label="Projector")
    for fc2 in frust_world2:
        ax2_3d.plot([pos2[0], fc2[0]], [pos2[1], fc2[1]], [pos2[2], fc2[2]],
                    "r-", alpha=0.5, lw=1)
    # Sync X and Y axis ranges so the 3D view doesn't jump or clip as sliders move
    _all_x2 = np.concatenate([GX2.ravel(), [pos2[0]], frust_world2[:, 0]])
    _all_y2 = np.concatenate([GY2.ravel(), [pos2[1]], frust_world2[:, 1]])
    _all_z2 = np.concatenate([GZ2.ravel(), [pos2[2]], frust_world2[:, 2]])
    _xy_half = max(_all_x2.max() - _all_x2.min(), _all_y2.max() - _all_y2.min()) / 2 * 1.3
    _x_mid2 = (_all_x2.max() + _all_x2.min()) / 2
    _y_mid2 = (_all_y2.max() + _all_y2.min()) / 2
    _z_pad2 = (_all_z2.max() - _all_z2.min()) * 0.1
    ax2_3d.set_xlim(_x_mid2 - _xy_half, _x_mid2 + _xy_half)
    ax2_3d.set_ylim(_y_mid2 - _xy_half, _y_mid2 + _xy_half)
    ax2_3d.set_zlim(_all_z2.min() - _z_pad2, _all_z2.max() + _z_pad2)
    ax2_3d.set_xlabel("X (mm)", fontsize=8)
    ax2_3d.set_ylabel("Y (mm)", fontsize=8)
    ax2_3d.set_zlabel("Z (mm)", fontsize=8)
    ax2_3d.set_title("3D scene + camera frustum", fontsize=9)
    ax2_3d.legend(fontsize=8)

    ax2_2d = fig2.add_subplot(1, 2, 2)
    ax2_2d.set_facecolor("#111")
    ax2_2d.scatter(u2[valid2], v2[valid2], c=GZ2.ravel()[valid2], cmap="terrain", s=25, zorder=3)
    ax2_2d.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=1.5))
    ax2_2d.set_xlim(0, W); ax2_2d.set_ylim(H, 0)
    ax2_2d.set_xlabel("Pixel x", color="white")
    ax2_2d.set_ylabel("Pixel y", color="white")
    ax2_2d.tick_params(colors="white")
    ax2_2d.set_title("Projected terrain", color="white", fontsize=9)
    plt.tight_layout()
    return (fig2,)


@app.cell
def _(cam_dx_s, cam_dy_s, cam_dz_s, cam_pitch_s, cam_yaw_s, fig2, mo):
    mo.vstack([
        mo.vstack([cam_dx_s, cam_dy_s, cam_dz_s, cam_yaw_s, cam_pitch_s]),
        fig2,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Lens Distortion

    Real projector lenses don't project perfectly straight lines. The glass bends the image in a
    pattern called **radial distortion**, which grows from zero at the centre outward to the edges.

    - **Barrel distortion** (k1 < 0): lines bow outward — the image bulges like a barrel
    - **Pincushion distortion** (k1 > 0): lines bow inward — the corners pull in

    The model corrects for this by modifying the normalised coordinates *before* applying focal
    length and principal point. Let $r^2 = x_n^2 + y_n^2$ be the squared distance from the
    optical axis in normalised space. Then:

    $$\text{scale} = 1 + k_1 r^2 + k_2 r^4$$
    $$x_\text{corrected} = x_n \cdot \text{scale} \qquad y_\text{corrected} = y_n \cdot \text{scale}$$

    Two coefficients (k1, k2) can model both primary and secondary radial effects. In the
    calibrated icefield projector k1 is small (< 0.01), but even a small value shifts edge pixels
    noticeably.

    > **In the code:** `geometric.py` → `project_points()` lines 340–346
    """)
    return


@app.cell
def _(mo):
    k1_s = mo.ui.slider(-0.50, 0.50, step=0.02, value=0.00, label="k1 (primary radial)")
    k2_s = mo.ui.slider(-0.10, 0.10, step=0.005, value=0.00, label="k2 (secondary radial)")
    return k1_s, k2_s


@app.cell
def _(k1_s, k2_s, np, plt):
    k1v = k1_s.value
    k2v = k2_s.value

    grid_lines3 = np.linspace(-0.9, 0.9, 9)
    t3 = np.linspace(-1.0, 1.0, 100)

    def _distort3(xn, yn, k1, k2):
        r2 = xn**2 + yn**2
        s = 1 + k1 * r2 + k2 * r2**2
        return xn * s, yn * s

    fig3, (ax3L, ax3R) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig3.patch.set_facecolor("#1a1a2e")

    for ax3, kk1, kk2, title3 in [
        (ax3L, 0.0, 0.0, "Undistorted"),
        (ax3R, k1v, k2v, f"k1 = {k1v:.2f},  k2 = {k2v:.3f}"),
    ]:
        ax3.set_facecolor("#111")
        ax3.set_aspect("equal")
        ax3.set_xlim(-1.3, 1.3); ax3.set_ylim(-1.3, 1.3)
        ax3.tick_params(colors="white")
        ax3.set_title(title3, color="white", fontsize=10)
        for yc3 in grid_lines3:
            xd, yd = _distort3(t3, np.full_like(t3, yc3), kk1, kk2)
            ax3.plot(xd, yd, "c-", lw=0.9, alpha=0.8)
        for xc3 in grid_lines3:
            xd, yd = _distort3(np.full_like(t3, xc3), t3, kk1, kk2)
            ax3.plot(xd, yd, "c-", lw=0.9, alpha=0.8)
        ax3.plot(0, 0, "r+", ms=10, mew=2, zorder=5)

    plt.tight_layout()
    return (fig3,)


@app.cell
def _(fig3, k1_s, k2_s, mo):
    mo.vstack([mo.hstack([k1_s, k2_s]), fig3])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 4. The 3D Surface: From Geography to Model Space

    The projection doesn't land on a flat screen — it lands on a **3D-printed terrain model**.
    Every point on that surface corresponds to a geographic location (easting/northing) and
    elevation. Relating these two coordinate systems is what makes the whole thing work.

    The pipeline has four ingredients:

    1. **Bounding box shapefile** — defines which rectangle on Earth the model represents.
       Together with the physical model dimensions in mm, this gives us horizontal scale factors:
       $$s_x = W_\text{model} / X_\text{geo}, \quad s_y = H_\text{model} / Y_\text{geo}$$

    2. **DEM (digital elevation model)** — a raster of elevation in metres at each geographic grid
       point. Every pixel in the source imagery has a corresponding 3D position.

    3. **Vertical scale** — the model compresses ~200 m of real elevation into ~30 mm of plastic.
       The scale factor is applied uniformly: $s_z = D_\text{model} / \Delta Z_\text{geo}$.

    4. **Z-exaggeration** — to make ridges and valleys visible on the physical model, the vertical
       scale is artificially amplified by a multiplier. Without it the terrain looks almost flat:
       $$z_\text{model} = (z_\text{geo} - z_\text{min}) \cdot s_z \cdot z_\text{exaggeration}$$

    Every 3D point fed into the camera model is computed this way. The DEM grid becomes the
    input point cloud that is projected, then used to build the inverse remap for the final image.

    > **In the code:** `geometric.py` → `IcefieldProjectionMapper.geo_to_stl()` (lines 293–318)
    """)
    return


@app.cell
def _(mo):
    z_exag_s = mo.ui.slider(1.0, 6.0, step=0.25, value=2.0, label="Z exaggeration")
    return (z_exag_s,)


@app.cell
def _(mo, np, plt, z_exag_s):
    z_ex = z_exag_s.value
    rng4 = np.random.default_rng(0)

    geo_x4 = np.linspace(0, 5000.0, 40)
    geo_y4 = np.linspace(0, 5000.0, 40)
    GX4, GY4 = np.meshgrid(geo_x4, geo_y4)

    geo_z4 = (1400
              + 100 * np.exp(-((GX4 - 2500)**2 + (GY4 - 2000)**2) / 1_200_000)
              + 60 * np.sin(GX4 / 1000) * np.cos(GY4 / 1200)
              + 18 * rng4.standard_normal((40, 40)))
    geo_z_min4 = float(geo_z4.min())
    geo_z_range4 = float(geo_z4.max() - geo_z_min4)

    model_d4 = 30.0
    sz4 = model_d4 / geo_z_range4
    stl_z4 = (geo_z4 - geo_z_min4) * sz4 * z_ex

    mid4 = 20
    geo_cross4 = geo_z4[mid4]
    stl_cross4 = stl_z4[mid4]
    stl_x_cross4 = np.linspace(0, 200.0, 40)

    fig4, (ax4a, ax4b) = plt.subplots(1, 2, figsize=(11, 4))
    fig4.patch.set_facecolor("#1a1a2e")

    ax4a.set_facecolor("#111")
    ax4a.plot(geo_x4 / 1000, geo_cross4, "c-", lw=1.5)
    ax4a.fill_between(geo_x4 / 1000, geo_z_min4, geo_cross4, alpha=0.25, color="cyan")
    ax4a.set_xlabel("Easting (km)", color="white")
    ax4a.set_ylabel("Elevation (m)", color="white")
    ax4a.set_title("Geographic cross-section (real metres)", color="white", fontsize=9)
    ax4a.tick_params(colors="white")

    ax4b.set_facecolor("#111")
    ax4b.plot(stl_x_cross4, stl_cross4, color="orange", lw=1.5)
    ax4b.fill_between(stl_x_cross4, 0, stl_cross4, alpha=0.25, color="orange")
    ax4b.axhline(model_d4, color="gray", ls="--", lw=0.8, alpha=0.6,
                 label=f"Nominal model top ({model_d4} mm)")
    ax4b.set_xlabel("Model X (mm)", color="white")
    ax4b.set_ylabel("Model Z (mm)", color="white")
    ax4b.set_title(f"Model cross-section  (z_exag = {z_ex:.2f}×)", color="white", fontsize=9)
    ax4b.tick_params(colors="white")
    ax4b.set_ylim(0, max(model_d4 * 7, stl_cross4.max() * 1.25))
    ax4b.legend(facecolor="#333", labelcolor="white", fontsize=8)
    plt.tight_layout()

    note4 = mo.callout(
        mo.md(
            f"At z_exaggeration = **{z_ex:.2f}×**, {geo_z_range4:.0f} m of real relief becomes "
            f"**{(geo_z_range4 * sz4 * z_ex):.1f} mm** on the model.  "
            f"Without exaggeration the terrain would only be **{geo_z_range4 * sz4:.1f} mm** tall — "
            f"nearly invisible."
        ),
        kind="info",
    )
    return fig4, note4


@app.cell
def _(fig4, mo, note4, z_exag_s):
    mo.vstack([z_exag_s, fig4, note4])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. Ground Control Points and Pose Optimisation

    We now have a complete camera model and a 3D terrain surface. The remaining question is:
    *what is the projector's actual pose?*

    The projector is a physical device mounted above the model. Its position can only be measured
    crudely with a ruler. To get pixel-accurate alignment we use **ground control points (GCPs)**.

    A GCP is a reference location whose 3D model-space coordinates are known (e.g., a corner stake,
    a geographic landmark) and whose *projected pixel position* can be measured in the image — the
    operator drags a cursor to mark each GCP on screen.

    The camera model lets us *predict* where a GCP should appear given any candidate pose.
    **Reprojection error** is the distance between the prediction and the observation:

    $$e_i = \hat{p}_i(\text{pose}) - p_i^\text{obs}$$

    Calibration minimises the total squared reprojection error across all N GCPs:

    $$\min_\text{pose} \sum_{i=1}^N \left\|e_i\right\|^2$$

    This nonlinear least-squares problem is solved with **Levenberg-Marquardt** — an algorithm
    that blends gradient descent with a linear approximation and converges reliably even from a
    rough starting point. The optimised parameters are: position (3 DOF), look direction via
    Rodrigues rotation (3 DOF), and throw ratio (1 DOF) — seven in total.

    > **In the code:** `tuning.py` → `_compute_reprojection_residuals()` (lines 829–862)
    > `tuning.py` → `_run_optimization()` (lines 864–932)

    The slider adds a pose perturbation. The right panel always shows the result after the
    optimiser runs from that perturbed starting point.
    """)
    return


@app.cell
def _(mo):
    perturb_s = mo.ui.slider(0.0, 1.0, step=0.05, value=0.3, label="Pose perturbation magnitude")
    return (perturb_s,)


@app.cell
def _(H, W, least_squares, np, perturb_s, plt, tr):
    rng5 = np.random.default_rng(7)

    gt_pos5 = np.array([100.0, -60.0, 430.0])
    gt_fwd5 = np.array([0.0, 0.18, -1.0]) / np.linalg.norm([0.0, 0.18, -1.0])
    gt_up5 = np.array([0.0, 1.0, 0.0])
    gt_tr5 = tr

    def _view5(pos, fwd, up_v):
        fwd = fwd / np.linalg.norm(fwd)
        r5 = np.cross(fwd, up_v); r5 /= np.linalg.norm(r5)
        u5 = np.cross(r5, fwd)
        R5 = np.array([r5, -u5, fwd])
        return R5, -R5 @ pos

    def _project5(pts, pos, fwd, up_v, throw):
        R5p, t5p = _view5(pos, fwd, up_v)
        cam5 = (R5p @ pts.T).T + t5p
        ok5 = cam5[:, 2] > 0
        f5 = W * throw
        pu = np.where(ok5, f5 * cam5[:, 0] / cam5[:, 2] + W / 2, np.nan)
        pv = np.where(ok5, f5 * cam5[:, 1] / cam5[:, 2] + H / 2, np.nan)
        return pu, pv

    gcp5 = np.array([
        [20,  20,  0], [100,  20, 0], [180,  20, 0],
        [20, 100,  5], [180, 100, 5],
        [20, 180,  0], [100, 180, 0], [180, 180, 0],
    ], dtype=float)

    u5_gt, v5_gt = _project5(gcp5, gt_pos5, gt_fwd5, gt_up5, gt_tr5)
    obs5 = np.stack([u5_gt, v5_gt], axis=1) + rng5.normal(0, 2.0, (len(gcp5), 2))

    p5 = perturb_s.value

    # Zoom (throw_ratio) as the primary perturbation: every projected point
    # shifts radially outward from the principal point by an amount proportional
    # to its distance from center, so GCPs in different image quadrants get
    # arrows pointing in different directions. The optimizer corrects throw_ratio
    # as one of its 7 DOF.
    # A small lateral position offset breaks the perfect radial symmetry to
    # better resemble a realistic miscalibration.
    pert_tr5 = gt_tr5 * (1.0 + 0.30 * p5)
    pert_pos5 = gt_pos5 + p5 * np.array([30.0, 0.0, 0.0])
    pert_fwd5 = gt_fwd5

    u5_pert, v5_pert = _project5(gcp5, pert_pos5, pert_fwd5, gt_up5, pert_tr5)
    err_pert = float(np.nanmean(np.sqrt((u5_pert - obs5[:, 0])**2 + (v5_pert - obs5[:, 1])**2)))

    def _residuals5(vec):
        ur, vr = _project5(gcp5, vec[:3], vec[3:6], gt_up5, vec[6])
        dxr = np.where(np.isfinite(ur), ur - obs5[:, 0], 1e4)
        dyr = np.where(np.isfinite(vr), vr - obs5[:, 1], 1e4)
        return np.concatenate([dxr, dyr])

    res5 = least_squares(_residuals5, np.concatenate([pert_pos5, pert_fwd5, [pert_tr5]]), method="lm")
    u5_opt, v5_opt = _project5(gcp5, res5.x[:3], res5.x[3:6], gt_up5, res5.x[6])
    err_opt = float(np.nanmean(np.sqrt((u5_opt - obs5[:, 0])**2 + (v5_opt - obs5[:, 1])**2)))

    fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(11, 4.5))
    fig5.patch.set_facecolor("#1a1a2e")

    for ax5, u5p, v5p, title5, err5 in [
        (ax5a, u5_pert, v5_pert, "Perturbed pose (before optimiser)", err_pert),
        (ax5b, u5_opt,  v5_opt,  "After Levenberg-Marquardt",         err_opt),
    ]:
        ax5.set_facecolor("#111")
        ax5.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=1))
        ax5.set_xlim(0, W); ax5.set_ylim(H, 0)
        ax5.tick_params(colors="white")
        ax5.set_title(f"{title5}\nmean reprojection error = {err5:.1f} px",
                      color="white", fontsize=9)
        ax5.scatter(obs5[:, 0], obs5[:, 1], color="lime", s=55, zorder=5, label="Observed (operator)")
        ax5.scatter(u5p, v5p, color="red", marker="x", s=65, lw=1.5, zorder=5, label="Predicted (model)")
        for i5 in range(len(gcp5)):
            if np.isfinite(u5p[i5]):
                ax5.annotate("", xy=(u5p[i5], v5p[i5]), xytext=(obs5[i5, 0], obs5[i5, 1]),
                             arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2))
        ax5.legend(facecolor="#333", labelcolor="white", fontsize=8)

    plt.tight_layout()
    return (fig5,)


@app.cell
def _(fig5, mo, perturb_s):
    mo.vstack([perturb_s, fig5])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 6. Polynomial Residual Correction

    Even after a successful pose optimisation, small residuals persist at the control points.
    These come from sources the camera model can't capture:

    - Physical imperfections in the 3D print (surface bumps, slight warping from cooling)
    - Measurement noise from the operator's cursor placement
    - Higher-order optical effects beyond k1/k2

    Rather than expanding the optical model further, we fit a **degree-2 polynomial warp** that
    absorbs the remaining error. The warp is expressed in the 6-term basis:

    $$\mathbf{a} = [1,\ x,\ y,\ x^2,\ xy,\ y^2]$$

    Separate polynomials are solved for x and y corrections via least squares:

    $$\mathbf{a}_i \cdot \mathbf{c}_x \approx \Delta x_i \qquad
      \mathbf{a}_i \cdot \mathbf{c}_y \approx \Delta y_i$$

    The 6 coefficients for each axis are stored in `ProjectorConfig` as `poly2_coeff_x` and
    `poly2_coeff_y`. At render time they define a dense pixel-by-pixel correction map applied as
    a second `cv2.remap` pass.

    > **In the code:** `tuning.py` → `_update_correction()` (lines 658–703)

    Toggle the switch to see residuals at control points before and after the polynomial fit,
    plus the correction surface that was learned.
    """)
    return


@app.cell
def _(mo):
    coeff_x6_state, set_coeff_x6 = mo.state([0.0] * 6)
    coeff_y6_state, set_coeff_y6 = mo.state([0.0] * 6)
    return coeff_x6_state, coeff_y6_state, set_coeff_x6, set_coeff_y6


@app.cell
def _(H, W, lstsq, np):
    rng6 = np.random.default_rng(13)
    n6 = 14

    ctrl_x6 = rng6.uniform(60, W - 60, n6)
    ctrl_y6 = rng6.uniform(40, H - 40, n6)

    # Normalised coordinates so all polynomial coefficients live on a similar pixel scale
    ox6, oy6 = W / 2.0, H / 2.0
    xn6 = (ctrl_x6 - ox6) / (W / 2.0)
    yn6 = (ctrl_y6 - oy6) / (H / 2.0)

    # Synthetic residuals: radial expansion + mild shear + noise
    dx6_raw = 2.5 * xn6 + 0.6 * yn6 + rng6.normal(0, 0.7, n6)
    dy6_raw = 0.4 * xn6 + 2.1 * yn6 + rng6.normal(0, 0.7, n6)

    A6 = np.column_stack([np.ones(n6), xn6, yn6, xn6 ** 2, xn6 * yn6, yn6 ** 2])
    opt_cx6, _, _, _ = lstsq(A6, dx6_raw)
    opt_cy6, _, _, _ = lstsq(A6, dy6_raw)
    return (
        A6,
        ctrl_x6,
        ctrl_y6,
        dx6_raw,
        dy6_raw,
        n6,
        opt_cx6,
        opt_cy6,
        ox6,
        oy6,
    )


@app.cell
def _(coeff_x6_state, coeff_y6_state, mo):
    _cx_init = coeff_x6_state()
    _cy_init = coeff_y6_state()
    _labels = ["const", "x", "y", "x²", "xy", "y²"]

    sliders_x6 = mo.ui.array(
        [mo.ui.slider(-20.0, 20.0, value=float(_cx_init[i]), step=0.05, label=f"cx  [{_labels[i]}]")
         for i in range(6)],
        label="X-correction coefficients",
    )
    sliders_y6 = mo.ui.array(
        [mo.ui.slider(-20.0, 20.0, value=float(_cy_init[i]), step=0.05, label=f"cy  [{_labels[i]}]")
         for i in range(6)],
        label="Y-correction coefficients",
    )
    learn_btn_6 = mo.ui.run_button(label="Learn optimal coefficients from GCPs")
    return learn_btn_6, sliders_x6, sliders_y6


@app.cell
def _(learn_btn_6, opt_cx6, opt_cy6, set_coeff_x6, set_coeff_y6):
    if learn_btn_6.value:
        set_coeff_x6(opt_cx6.tolist())
        set_coeff_y6(opt_cy6.tolist())
    return


@app.cell
def _(
    A6,
    H,
    W,
    ctrl_x6,
    ctrl_y6,
    dx6_raw,
    dy6_raw,
    mo,
    n6,
    np,
    ox6,
    oy6,
    plt,
    sliders_x6,
    sliders_y6,
):
    cx6 = np.array([s.value for s in sliders_x6])
    cy6 = np.array([s.value for s in sliders_y6])

    # Residuals after applying the current polynomial correction
    dx_pred6 = A6 @ cx6
    dy_pred6 = A6 @ cy6
    dx_res6 = dx6_raw - dx_pred6
    dy_res6 = dy6_raw - dy_pred6
    mean_res6 = float(np.mean(np.sqrt(dx_res6 ** 2 + dy_res6 ** 2)))

    # Dense warped grid for the warp-field visualisation
    nx_g, ny_g = 24, 14
    gx = np.linspace(0, W, nx_g)
    gy = np.linspace(0, H, ny_g)
    gxx, gyy = np.meshgrid(gx, gy)
    gxn = (gxx - ox6) / (W / 2.0)
    gyn = (gyy - oy6) / (H / 2.0)
    A_g6 = np.column_stack([
        np.ones(gxx.size), gxn.ravel(), gyn.ravel(),
        gxn.ravel() ** 2, (gxn * gyn).ravel(), gyn.ravel() ** 2,
    ])
    dx_g6 = (A_g6 @ cx6).reshape(gxx.shape)
    dy_g6 = (A_g6 @ cy6).reshape(gyy.shape)
    warped_gx = gxx + dx_g6
    warped_gy = gyy + dy_g6

    fig6, (ax6a, ax6b) = plt.subplots(1, 2, figsize=(11, 4))
    fig6.patch.set_facecolor("#1a1a2e")

    # Left panel: residual arrows
    ax6a.set_facecolor("#111")
    ax6a.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=1))
    ax6a.set_xlim(0, W)
    ax6a.set_ylim(H, 0)
    ax6a.tick_params(colors="white")
    ax6a.set_title(f"Residuals after poly2  |  mean = {mean_res6:.2f} px", color="white", fontsize=9)
    _scale6 = 20
    for _i6 in range(n6):
        ax6a.annotate(
            "",
            xy=(ctrl_x6[_i6] + dx_res6[_i6] * _scale6, ctrl_y6[_i6] + dy_res6[_i6] * _scale6),
            xytext=(ctrl_x6[_i6], ctrl_y6[_i6]),
            arrowprops=dict(arrowstyle="->", color="yellow", lw=1.2),
        )
    ax6a.scatter(ctrl_x6, ctrl_y6, color="lime", s=30, zorder=5)
    ax6a.set_xlabel("Pixel x", color="white")
    ax6a.set_ylabel("Pixel y", color="white")
    ax6a.text(5, H - 8, f"(arrows ×{_scale6})", color="gray", fontsize=7)

    # Right panel: displacement magnitude heatmap + warped grid overlay
    # Dense surface for the colour background
    nx_dense, ny_dense = 120, 70
    hx = np.linspace(0, W, nx_dense)
    hy = np.linspace(0, H, ny_dense)
    hxx, hyy = np.meshgrid(hx, hy)
    hxn = (hxx - ox6) / (W / 2.0)
    hyn = (hyy - oy6) / (H / 2.0)
    A_h6 = np.column_stack([
        np.ones(hxx.size), hxn.ravel(), hyn.ravel(),
        hxn.ravel() ** 2, (hxn * hyn).ravel(), hyn.ravel() ** 2,
    ])
    hdx = (A_h6 @ cx6).reshape(hxx.shape)
    hdy = (A_h6 @ cy6).reshape(hyy.shape)
    mag6 = np.sqrt(hdx ** 2 + hdy ** 2)
    vmax6 = max(float(mag6.max()), 0.5)

    ax6b.set_xlim(0, W)
    ax6b.set_ylim(H, 0)
    ax6b.tick_params(colors="white")
    ax6b.set_title("Correction warp field (displacement magnitude)", color="white", fontsize=9)
    im6 = ax6b.imshow(
        mag6, extent=[0, W, H, 0], cmap="plasma",
        vmin=0, vmax=vmax6, origin="upper", aspect="auto",
    )
    for _row in range(ny_g):
        ax6b.plot(warped_gx[_row], warped_gy[_row], color="white", lw=0.7, alpha=0.5)
    for _col in range(nx_g):
        ax6b.plot(warped_gx[:, _col], warped_gy[:, _col], color="white", lw=0.7, alpha=0.5)
    ax6b.set_xlabel("Pixel x", color="white")
    ax6b.set_ylabel("Pixel y", color="white")
    cbar6 = fig6.colorbar(im6, ax=ax6b)
    cbar6.set_label("|displacement| (px)", color="white")
    cbar6.ax.yaxis.set_tick_params(colors="white")

    plt.tight_layout()

    note6 = mo.callout(
        mo.md(
            "The polynomial captures the **systematic** part of the error. After correction, only "
            "random per-point noise remains — which cannot be reduced without better GCP measurements."
        ),
        kind="info",
    )
    return fig6, note6


@app.cell
def _(fig6, learn_btn_6, mo, note6, sliders_x6, sliders_y6):
    mo.vstack([
        mo.hstack([sliders_x6, sliders_y6], justify="start"),
        learn_btn_6,
        fig6,
        note6,
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 7. Image Draping and the Inverse Warp

    All the previous steps feed into a single goal: producing a pre-distorted image that, when
    projected onto the physical terrain model, looks as if the source raster is painted directly
    on the surface.

    ### Why forward projection alone isn't enough

    You might think the obvious approach is to forward-project every source pixel through the
    camera model to find its output location. In practice this leaves **gaps** (output pixels no
    source pixel maps to) and **overlaps** (many source pixels competing for the same output
    pixel). The result is a sparse, hole-filled image.

    ### The inverse-map pipeline

    Instead we build an *inverse* lookup table: for every output pixel $(u, v)$, which geographic
    raster coordinate should be sampled?

    **Step 1 — Forward-project the DEM grid.**
    Sample the DEM at regular geographic intervals, convert each sample to STL model coordinates,
    then project through the camera model. Each DEM grid point $(r_i, c_i)$ now has a
    corresponding pixel position $(u_i, v_i)$ in the output image.

    **Step 2 — Delaunay triangulate.**
    Triangulate the $(u_i, v_i)$ cloud in output-pixel space. Each resulting **simplex** (triangle
    in 2D) covers a small patch of the output image and has three DEM-index vertices. Together
    they tile the entire visible terrain surface.

    **Step 3 — Simplex filtering.**
    Triangles at the boundary of the valid terrain region can bridge across *gaps* — regions where
    there was no valid DEM data (outside the physical model's footprint, nodata cells, etc.).
    These bridging triangles connect DEM points that are far apart in grid space even though they
    project near each other in pixel space. They cause stretching artefacts and are discarded by
    checking the Manhattan distance between vertex grid indices:

    $$\max_{(a,b) \in \text{edges}}\!\bigl(|r_a - r_b| + |c_a - c_b|\bigr) > \text{threshold}
      \implies \text{reject triangle}$$

    **Step 4 — Barycentric interpolation.**
    For every output pixel inside a surviving triangle, compute its barycentric weights
    $(b_0, b_1, b_2)$ and interpolate the DEM grid index:

    $$r = b_0 r_0 + b_1 r_1 + b_2 r_2 \qquad c = b_0 c_0 + b_1 c_1 + b_2 c_2$$

    **Step 5 — `cv2.remap`.**
    The dense `map_row` / `map_col` arrays (one float per output pixel) are passed to OpenCV's
    `remap`, which samples the source image in a single vectorised pass.

    > **In the code:** `geometric.py` → `_build_inverse_map()` (lines 55–138),
    > `IcefieldProjectionMapper.compute_projection_map()` (lines 360–392)

    The synthetic terrain below has a **nodata gap** across the middle column range. Toggle the
    simplex filter to see what happens when bridging triangles are included versus removed.
    """)
    return


@app.cell
def _(mo):
    simplex_filter_s = mo.ui.switch(label="Simplex filter ON", value=True)
    return (simplex_filter_s,)


@app.cell
def _(H, W, np):
    from scipy.spatial import Delaunay as _Delaunay7

    ng_r7, ng_c7 = 15, 20
    _gx7 = np.linspace(-200, 200, ng_c7)
    _gy7 = np.linspace(-150, 150, ng_r7)
    gxx7, gyy7 = np.meshgrid(_gx7, _gy7)
    gz7 = 18 * np.sin(gxx7 / 100) * np.cos(gyy7 / 80)

    # Nodata gap: 4 columns in the centre simulating terrain outside the model footprint
    _gap7 = ng_c7 // 2
    valid7 = np.ones((ng_r7, ng_c7), dtype=bool)
    valid7[:, _gap7 - 2:_gap7 + 2] = False

    _ri7 = np.repeat(np.arange(ng_r7), ng_c7)
    _ci7 = np.tile(np.arange(ng_c7), ng_r7)

    # Camera: directly overhead with a slight forward tilt
    _look7 = np.array([0.0, 0.05, -1.0])
    _look7 /= np.linalg.norm(_look7)
    _up7 = np.array([0.0, 1.0, 0.05])
    _up7 /= np.linalg.norm(_up7)
    _right7 = np.cross(_look7, _up7)
    _right7 /= np.linalg.norm(_right7)
    _up_true7 = np.cross(_right7, _look7)
    _R7 = np.array([_right7, -_up_true7, _look7])
    _t7 = -_R7 @ np.array([0.0, 0.0, 450.0])
    _fx7 = W * 0.58

    _pts7 = np.stack([gxx7.ravel(), gyy7.ravel(), gz7.ravel()], axis=1)
    _pc7 = (_R7 @ _pts7.T).T + _t7
    _zc7 = _pc7[:, 2]
    _u7 = _fx7 * _pc7[:, 0] / _zc7 + W / 2
    _v7 = _fx7 * _pc7[:, 1] / _zc7 + H / 2

    _vp7 = valid7.ravel() & (_zc7 > 0) & (_u7 >= 0) & (_u7 < W) & (_v7 >= 0) & (_v7 < H)

    uv7 = np.stack([_u7[_vp7], _v7[_vp7]], axis=1)
    ri7 = _ri7[_vp7]
    ci7 = _ci7[_vp7]

    # Checkerboard source colour defined in DEM-grid space
    checker7 = ((ri7 // 2 + ci7 // 2) % 2).astype(float)

    # Delaunay triangulation of projected points
    _tri7 = _Delaunay7(uv7)
    simps7 = _tri7.simplices

    # Classify each simplex: bad if any edge spans more than 3 grid cells (Manhattan)
    _d01 = np.abs(ri7[simps7[:, 0]] - ri7[simps7[:, 1]]) + np.abs(ci7[simps7[:, 0]] - ci7[simps7[:, 1]])
    _d02 = np.abs(ri7[simps7[:, 0]] - ri7[simps7[:, 2]]) + np.abs(ci7[simps7[:, 0]] - ci7[simps7[:, 2]])
    _d12 = np.abs(ri7[simps7[:, 1]] - ri7[simps7[:, 2]]) + np.abs(ci7[simps7[:, 1]] - ci7[simps7[:, 2]])
    good7 = np.maximum(_d01, np.maximum(_d02, _d12)) <= 3
    return checker7, good7, simps7, uv7


@app.cell
def _(H, W, checker7, good7, mo, np, plt, simplex_filter_s, simps7, uv7):
    import matplotlib.patches as _mpatches7

    _use_filter = simplex_filter_s.value
    _good_s7 = simps7[good7]
    _bad_s7 = simps7[~good7]
    _active_s7 = _good_s7 if _use_filter else simps7

    fig7, (ax7a, ax7b) = plt.subplots(1, 2, figsize=(11, 4))
    fig7.patch.set_facecolor("#1a1a2e")

    # Left panel: Delaunay triangulation coloured by good / bad
    ax7a.set_facecolor("#111")
    ax7a.set_xlim(0, W)
    ax7a.set_ylim(H, 0)
    ax7a.tick_params(colors="white")
    ax7a.set_title(
        f"Delaunay in output-pixel space  ({int(good7.sum())} good / {int((~good7).sum())} bad simplices)",
        color="white", fontsize=9,
    )
    if len(_good_s7):
        ax7a.tripcolor(uv7[:, 0], uv7[:, 1], _good_s7,
                       np.ones(len(_good_s7)), shading="flat", cmap="Blues", vmin=0, vmax=1.5, alpha=0.25)
        ax7a.triplot(uv7[:, 0], uv7[:, 1], _good_s7, color="steelblue", lw=0.35, alpha=0.5)
    if len(_bad_s7):
        ax7a.tripcolor(uv7[:, 0], uv7[:, 1], _bad_s7,
                       np.ones(len(_bad_s7)), shading="flat", cmap="Reds", vmin=0, vmax=1.5, alpha=0.55)
        ax7a.triplot(uv7[:, 0], uv7[:, 1], _bad_s7, color="red", lw=0.5, alpha=0.7)
    ax7a.scatter(uv7[:, 0], uv7[:, 1], color="lime", s=7, zorder=5)
    ax7a.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=1))
    ax7a.set_xlabel("Output pixel u", color="white")
    ax7a.set_ylabel("Output pixel v", color="white")
    ax7a.legend(
        handles=[
            _mpatches7.Patch(facecolor="steelblue", alpha=0.5, label="Good (within gap threshold)"),
            _mpatches7.Patch(facecolor="red", alpha=0.6, label="Bad (bridges nodata gap)"),
        ],
        facecolor="#333", labelcolor="white", fontsize=8,
    )

    # Right panel: draped checkerboard image
    ax7b.set_facecolor("#111")
    ax7b.set_xlim(0, W)
    ax7b.set_ylim(H, 0)
    ax7b.tick_params(colors="white")
    ax7b.set_title(
        "Draped image — filter ON" if _use_filter else "Draped image — filter OFF  (gap bridged, artefacts visible)",
        color="white", fontsize=9,
    )
    ax7b.tripcolor(uv7[:, 0], uv7[:, 1], _active_s7, checker7, cmap="gray", vmin=0, vmax=1, shading="gouraud")
    ax7b.add_patch(plt.Rectangle((0, 0), W, H, fill=False, edgecolor="cyan", lw=1))
    ax7b.set_xlabel("Output pixel u", color="white")
    ax7b.set_ylabel("Output pixel v", color="white")

    plt.tight_layout()

    note7 = mo.callout(
        mo.md(
            "With the filter **off**, bad triangles bridge the nodata gap and stretch the "
            "checkerboard across it. With the filter **on**, the gap is a clean void — exactly "
            "what the projector casts where no valid terrain surface exists."
        ),
        kind="warn",
    )
    return fig7, note7


@app.cell
def _(fig7, mo, note7, simplex_filter_s):
    mo.vstack([simplex_filter_s, fig7, note7])
    return


if __name__ == "__main__":
    app.run()
