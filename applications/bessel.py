import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import jv, jn_zeros

# ---- พารามิเตอร์ ----
a = 1
T = 100
rho = 0.01
c = np.sqrt(T / rho)

m, n = 0,3
A = 0.05
phi = 0.0

alpha_mn = jn_zeros(m, n)[-1]
k_mn = alpha_mn / a
omega = c * k_mn
freq = omega / 2*np.pi
print(f"mode (m={m}, n={n}), alpha={alpha_mn:.6f}, omega={omega:.6f}, freq={freq:.6f}")

# ---- mesh ----
Nr, Nth = 80, 160
r = np.linspace(0, a, Nr)
theta = np.linspace(0, 2*np.pi, Nth)
R, Theta = np.meshgrid(r, theta, indexing='xy')
X = (R * np.cos(Theta)).T
Y = (R * np.sin(Theta)).T

def radial_profile(r_vals):
    return jv(m, k_mn * r_vals)

R_flat = np.sqrt(X**2 + Y**2)
Rad = radial_profile(R_flat)
Theta_xy = np.arctan2(Y, X)
angular_part = np.cos(m * Theta_xy)

def z_field(t):
    return A * Rad * angular_part * np.cos(omega * t + phi)

# ---- figure ----
fig = plt.figure(figsize=(7,5))
ax = fig.add_subplot(111, projection='3d')
ax.set_box_aspect((1,1,0.4))
ax.set_xlim(-a, a)
ax.set_ylim(-a, a)
ax.set_zlim(-A*1.2, A*1.2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
ax.view_init(elev=30, azim=120)

# initial surface
Z0 = z_field(0)
surf = ax.plot_surface(X, Y, Z0, rstride=2, cstride=2, cmap='viridis')

def update(frame):
    global surf
    # ลบ surface เก่า
    surf.remove()
    # วาด surface ใหม่
    Z = z_field(frame * 0.05)
    surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap='viridis')
    return surf,

anim = animation.FuncAnimation(fig, update, frames=60, interval=30, blit=False)
plt.show()
