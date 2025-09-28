# spoon_two_models_fixed.py
# ซ้าย: Spherical cap (ขอบกลม) / ขวา: Ellipsoidal cap (ขอบวงรี)
# เพิ่มกราฟความสัมพันธ์ z กับ h สำหรับตำแหน่งสัดส่วนรัศมี s หลายค่า

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator

# ---------------- helpers ----------------
def set_axes_equal(ax):
    """ทำให้สเกลแกน x,y,z เท่ากันเพื่อไม่ให้รูปดูบิด"""
    x_lim = ax.get_xlim3d(); y_lim = ax.get_ylim3d(); z_lim = ax.get_zlim3d()
    x_range = abs(x_lim[1] - x_lim[0]); x_mid = 0.5 * (x_lim[0] + x_lim[1])
    y_range = abs(y_lim[1] - y_lim[0]); y_mid = 0.5 * (y_lim[0] + y_lim[1])
    z_range = abs(z_lim[1] - z_lim[0]); z_mid = 0.5 * (z_lim[0] + z_lim[1])
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - R, x_mid + R])
    ax.set_ylim3d([y_mid - R, y_mid + R])
    ax.set_zlim3d([z_mid - R, z_mid + R])

def elliptical_polar_mesh(a, b, nr=220, nt=440):
    """
    กริดเชิงขั้ววงรี: s∈[0,1], θ∈[0,2π]
    s=1 จะได้ขอบวงรี x^2/a^2 + y^2/b^2 = 1
    """
    s = np.linspace(0.0, 1.0, nr)
    t = np.linspace(0.0, 2*np.pi, nt)
    S, T = np.meshgrid(s, t, indexing="ij")
    X = a * S * np.cos(T)
    Y = b * S * np.sin(T)
    return X, Y, S

# ---------------- models ----------------
def spherical_cap(a_rim, h, X, Y):
    """
    ฝาทรงกลม (ขอบรัศมี a_rim, ความลึก h)
    R = (a_rim^2 + h^2) / (2h)
    z = R - sqrt(R^2 - r^2), r^2 = x^2 + y^2, r<=a_rim
    """
    R = (a_rim**2 + h**2) / (2.0 * h)
    r2 = X**2 + Y**2
    Z = R - np.sqrt(np.maximum(R**2 - r2, 0.0))
    return Z, R

def ellipsoidal_cap(a, b, h, X, Y):
    """
    ฝาทรงรีแท้ (cap ของ ellipsoid): 
    z = h * (1 - sqrt(1 - x^2/a^2 - y^2/b^2)), สำหรับ x^2/a^2 + y^2/b^2 <= 1
    ศูนย์กลาง z=0, ขอบ z=h (แสดงเป็น "ความลึก" 0..h)
    """
    rho2 = (X**2)/(a**2) + (Y**2)/(b**2)
    Z = h * (1.0 - np.sqrt(np.maximum(1.0 - rho2, 0.0)))
    return Z

# ---------------- parameters (mm) ----------------
a = 25.0   # ครึ่งแกนวงรีตามแกน x (สำหรับขวา)
b = 20.0   # ครึ่งแกนวงรีตามแกน y (สำหรับขวาและใช้อ้างเทียบ)
h = 7.0    # ความลึกชาม
nr, nt = 220, 440

# ---------- สร้างกริด ----------
# ซ้าย (ขอบกลม): ใช้ a_circ = min(a,b) เพื่อให้เป็นวงกลมพอดี
a_circ = min(a, b)
Xc, Yc, _ = elliptical_polar_mesh(a_circ, a_circ, nr=nr, nt=nt)

# ขวา (ขอบวงรี)
Xe, Ye, _ = elliptical_polar_mesh(a, b, nr=nr, nt=nt)

# ---------- คำนวณพื้นผิว ----------
Zs, R_sphere = spherical_cap(a_circ, h, Xc, Yc)  # ซ้าย
Ze = ellipsoidal_cap(a, b, h, Xe, Ye)           # ขวา

# ---------- วาดรูป 3D (สไตล์เดียวกับตัวอย่าง) ----------
fig = plt.figure(figsize=(12, 8))

# ซ้าย: Spherical cap (ขอบกลม)
ax1 = fig.add_subplot(1, 1, 1, projection="3d")
surf1 = ax1.plot_surface(Xc, Yc, Zs, cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax1.set_title(f"Spherical cap (rim={a_circ:.1f} mm, h={h:.1f} mm, sphere R={R_sphere:.2f} mm)")
ax1.set_xlabel("x (mm)"); ax1.set_ylabel("y (mm)"); ax1.set_zlabel("z (mm)")
ax1.set_zlim(0.0, h)
ax1.zaxis.set_major_locator(LinearLocator(10))
ax1.zaxis.set_major_formatter('{x:.02f}')
fig.colorbar(surf1, ax=ax1, shrink=0.6, aspect=12)
set_axes_equal(ax1)
ax1.view_init(elev=30, azim=-60)

# ขวา: Ellipsoidal cap (ขอบวงรี)
#ax2 = fig.add_subplot(2, 2, 2, projection="3d")
#surf2 = ax2.plot_surface(Xe, Ye, Ze, cmap=cm.coolwarm, linewidth=0, antialiased=False)
#ax2.set_title(f"Ellipsoidal cap (a={a:.1f} mm, b={b:.1f} mm, h={h:.1f} mm)")
#ax2.set_xlabel("x (mm)"); ax2.set_ylabel("y (mm)"); ax2.set_zlabel("z (mm)")
#ax2.set_zlim(0.0, h)
#ax2.zaxis.set_major_locator(LinearLocator(10))
#ax2.zaxis.set_major_formatter('{x:.02f}')
#fig.colorbar(surf2, ax=ax2, shrink=0.6, aspect=12)
#set_axes_equal(ax2)
#ax2.view_init(elev=30, azim=-60)


plt.tight_layout()
plt.show()
