# 3D Quadratic Spoon Models (concave & convex) — complete, editable demo
# You can change the parameters in the "SET PARAMETERS" block below and re-run.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import cm
# =============== SET PARAMETERS ===============
# Geometry of the rim / curvature scale
a = 3.0     # semi-axis along x (cm)
b = 2.0     # semi-axis along y (cm)
h = 1.5     # target apex height for concave model (cm)

# Linear "tilt" terms (slope near the origin)
s_x = 0.1  # slope in x-direction
s_y = -0.25 # slope in y-direction

# Baseline for convex model (minimum level)
r0 = 0.4    # z at the unconstrained minimum for convex model

# Mesh resolution for plotting
nx = ny = 70
# =============================================


# ---------- Utilities ----------
def summarize_matrix(M):
    return np.array2string(M, precision=6, suppress_small=True)


# ---------- Concave (flipped spoon bowl) ----------
# z = h(1 - x^2/a^2 - y^2/b^2) + s_x x + s_y y on the ellipse x^2/a^2 + y^2/b^2 <= 1
def concave_analyze(a, b, h, s_x, s_y):
    # Interior critical point
    x_star = (s_x * a**2) / (2*h)
    y_star = (s_y * b**2) / (2*h)
    re2 = (x_star**2)/a**2 + (y_star**2)/b**2
    S = a**2 * s_x**2 + b**2 * s_y**2  # s^T M^{-1} s
    inside = re2 <= 1.0 + 1e-12

    if inside:
        z_star = h + S/(4*h)  # interior maximum value
        where_max = ("interior", (x_star, y_star, z_star))
        z_max = z_star
    else:
        amp = sqrt(S)
        # Location on the rim where maximum is attained
        if amp > 0:
            xb = (a**2 * s_x) / amp
            yb = (b**2 * s_y) / amp
        else:
            xb = 0.0; yb = 0.0
        where_max = ("rim", (xb, yb, amp))
        z_max = amp

    # Hessian and eigenvalues
    H = np.array([[-2*h/a**2, 0.0],
                  [0.0, -2*h/b**2]])
    eigvals = np.linalg.eigvalsh(H)

    # Build summary table
    summary = pd.DataFrame({
        "Item": [
            "Model", "Hessian H", "eig(H)",
            "Interior x*", "Interior y*", "r_e^2 at x*,y*",
            "S = a^2 s_x^2 + b^2 s_y^2", "Inside ellipse?",
            "z_max (value)", "Where z_max occurs"
        ],
        "Value": [
            "z = h(1 - x^2/a^2 - y^2/b^2) + s_x x + s_y y  with  x^2/a^2 + y^2/b^2 ≤ 1",
            summarize_matrix(H), eigvals.round(6),
            round(x_star, 6), round(y_star, 6), round(re2, 6),
            round(S, 6), inside,
            round(z_max, 6), where_max[0]
        ]
    })
    if where_max[0] == "interior":
        summary.loc[len(summary)] = ["argmax (x,y,z)", (x_star, y_star, z_max)]
    else:
        summary.loc[len(summary)] = ["argmax (x,y,z) on rim", (where_max[1][0], where_max[1][1], where_max[1][2])]
    return H, eigvals, z_max, where_max, summary


def concave_plot(a, b, h, s_x, s_y, where_max):
    # Build masked grid over the ellipse
    x = np.linspace(-a, a, nx)
    y = np.linspace(-b, b, ny)
    X, Y = np.meshgrid(x, y)
    mask = (X**2)/(a**2) + (Y**2)/(b**2) <= 1.0
    Z = np.full_like(X, np.nan, dtype=float)
    Z[mask] = h*(1 - (X[mask]**2)/(a**2) - (Y[mask]**2)/(b**2)) + s_x*X[mask] + s_y*Y[mask]

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, rstride=2, cstride=2, cmap=cm.viridis)

    # Plot rim (z=0 level for the quadratic core)
    theta = np.linspace(0, 2*np.pi, 500)
    xr = a*np.cos(theta); yr = b*np.sin(theta); zr = np.zeros_like(theta)
    ax.plot(xr, yr, zr, linewidth=0.7)

    # Mark the maximum
    tag, pt = where_max
    if tag == "interior":
        ax.scatter([pt[0]], [pt[1]], [pt[2]], s=60, color='red')
    elif tag == "rim":
        ax.scatter([pt[0]], [pt[1]], [pt[2]], s=60, color='red')

    ax.set_title(f"Concave quadratic spoon (domain: ellipse) a:{a} b:{b} h:{h} Sx:{s_x} Sy:{s_y}")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    ax.set_box_aspect((a, b, h))
    plt.show()


# ---------- Convex (upward elliptic paraboloid) ----------
# z = x^2/a^2 + y^2/b^2 + s_x x + s_y y + r0    (unconstrained)
def convex_analyze(a, b, s_x, s_y, r0):
    H = np.array([[2/a**2, 0.0],
                  [0.0, 2/b**2]])
    g = np.array([s_x, s_y])

    # Critical point u* = -H^{-1} g
    u_star = -np.linalg.solve(H, g)
    x_star, y_star = float(u_star[0]), float(u_star[1])
    # Value at the minimum
    z_star = 0.5*u_star @ H @ u_star + g @ u_star + r0

    eigvals = np.linalg.eigvalsh(H)

    summary = pd.DataFrame({
        "Item": ["Model", "Hessian H", "eig(H)", "x* (min)", "y* (min)", "z_min at (x*,y*)"],
        "Value": [
            "z = x^2/a^2 + y^2/b^2 + s_x x + s_y y + r0  (unconstrained)",
            summarize_matrix(H), eigvals.round(6),
            round(x_star, 6), round(y_star, 6), round(float(z_star), 6)
        ]
    })
    return H, eigvals, (x_star, y_star, float(z_star)), summary


def convex_plot(a, b, s_x, s_y, r0, star):
    # Grid for plotting
    x = np.linspace(-4*a/3, 4*a/3, nx)
    y = np.linspace(-4*b/3, 4*b/3, ny)
    X, Y = np.meshgrid(x, y)
    Z = (X**2)/(a**2) + (Y**2)/(b**2) + s_x*X + s_y*Y + r0

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True, rstride=3, cstride=3, cmap=cm.viridis)

    # Mark the minimum
    ax.scatter([star[0]], [star[1]], [star[2]], s=60, color='red')
    theta = np.linspace(0, 2*np.pi, 500)
    xr = a*np.cos(theta); yr = b*np.sin(theta); zr = np.zeros_like(theta)
    ax.plot(xr, yr, zr, linewidth=0.7)
    ax.set_title("Convex quadratic (unconstrained minimum)")
    ax.set_xlabel("x (cm)")
    ax.set_ylabel("y (cm)")
    ax.set_zlabel("z (cm)")
    plt.show()


# ---------- Run analyses ----------
Hc, evals_c, zmax_c, where_max_c, concave_tbl = concave_analyze(a, b, h, s_x, s_y)
concave_plot(a, b, h, s_x, s_y, where_max_c)

Hk, evals_k, star_k, convex_tbl = convex_analyze(a, b, s_x, s_y, r0)
convex_plot(a, b, s_x, s_y, r0, star_k)
