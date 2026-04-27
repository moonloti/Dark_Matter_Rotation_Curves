# --- Imports ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Load Data ---
df_3198 = pd.read_csv('data/NGC3198_rotmod.dat', sep='\s+', comment='#',
                      names=['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul'])

df_6503 = pd.read_csv('data/NGC6503_rotmod.dat', sep='\s+', comment='#',
                       names=['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul'])

df_ddo = pd.read_csv('data/DDO154_rotmod.dat', sep='\s+', comment='#',
                      names=['Rad','Vobs','errV','Vgas','Vdisk','Vbul','SBdisk','SBbul'])

# --- Clean Data ---
df_3198 = df_3198.dropna()
df_6503 = df_6503.dropna()
df_ddo  = df_ddo.dropna()

print("NGC 3198:")
print(df_3198.head())
print("\nNGC 6503:")
print(df_6503.head())
print("\nDDO 154:")
print(df_ddo.head())

# --- Calculate Visible Velocity ---
# combining gas, disk, bulge components
def calc_visible_velocity(df):
    vvis_sq = (np.sign(df['Vgas']) * df['Vgas']**2 +
               np.sign(df['Vdisk']) * df['Vdisk']**2 +
               np.sign(df['Vbul']) * df['Vbul']**2)
    df['Vvis'] = np.sqrt(np.abs(vvis_sq))
    return df

df_3198 = calc_visible_velocity(df_3198)
df_6503 = calc_visible_velocity(df_6503)
df_ddo  = calc_visible_velocity(df_ddo)

# --- Calculate Dark Matter Mass ---
# G in units compatible with SPARC data (pc, km/s, solar masses)
G = 4.302e-3  # pc * Msun^-1 * (km/s)^2

def calc_dark_matter(df, name):
    # Use outermost radius point
    r_out_pc = df['Rad'].iloc[-1] * 1000   # kpc to pc
    v_obs_out = df['Vobs'].iloc[-1]         # km/s
    v_vis_out = df['Vvis'].iloc[-1]         # km/s

    M_total   = (v_obs_out**2 * r_out_pc) / G
    M_visible = (v_vis_out**2 * r_out_pc) / G
    M_dark    = M_total - M_visible
    dm_frac   = M_dark / M_total

    print(f"\n--- {name} ---")
    print(f"Total mass:    {M_total:.3e} solar masses")
    print(f"Visible mass:  {M_visible:.3e} solar masses")
    print(f"Dark matter:   {M_dark:.3e} solar masses")
    print(f"DM fraction:   {dm_frac*100:.1f}%")

    return M_total, M_visible, M_dark, dm_frac

M_tot_3198, M_vis_3198, M_dark_3198, frac_3198 = calc_dark_matter(df_3198, "NGC 3198")
M_tot_6503, M_vis_6503, M_dark_6503, frac_6503 = calc_dark_matter(df_6503, "NGC 6503")
M_tot_ddo,  M_vis_ddo,  M_dark_ddo,  frac_ddo  = calc_dark_matter(df_ddo,  "DDO 154")

# --- Plot 1: Rotation Curves ---
# shows observed velocity vs Keplerian prediction for all 3 galaxies
fig, axes = plt.subplots(1, 3, figsize=(14, 5))

galaxies = [
    (df_3198, "NGC 3198", "steelblue"),
    (df_6503, "NGC 6503", "green"),
    (df_ddo,  "DDO 154",  "purple"),
]

for ax, (df, name, color) in zip(axes, galaxies):
    ax.scatter(df['Rad'], df['Vobs'], color=color, s=20,
               label='Observed', zorder=3)
    ax.plot(df['Rad'], df['Vvis'], '--', color='red',
            linewidth=2, label='Keplerian (visible only)')
    ax.fill_between(df['Rad'], df['Vvis'], df['Vobs'],
                    alpha=0.15, color=color)
    ax.set_title(name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Radius (kpc)')
    ax.set_ylabel('Velocity (km/s)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Observed vs. Keplerian Rotation Curves', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/rotation_curves.png', dpi=150)
plt.show()
print("Plot 1 saved!")

# --- Plot 2: Visible vs Total Mass Bar Chart ---
names    = ['NGC 3198', 'NGC 6503', 'DDO 154']
m_vis    = [M_vis_3198, M_vis_6503, M_vis_ddo]
m_tot    = [M_tot_3198, M_tot_6503, M_tot_ddo]
colors   = ['steelblue', 'green', 'purple']

x = np.arange(len(names))
w = 0.35

fig, ax = plt.subplots(figsize=(7, 5))
ax.bar(x - w/2, np.log10(m_vis), w, label='Visible Mass',
       color=colors, alpha=0.5)
ax.bar(x + w/2, np.log10(m_tot), w, label='Estimated Total Mass',
       color=colors, alpha=1.0)
ax.set_xticks(x)
ax.set_xticklabels(names)
ax.set_ylabel('log₁₀(Mass / M☉)')
ax.set_title('Visible vs. Estimated Total Mass per Galaxy', fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('figures/mass_comparison.png', dpi=150)
plt.show()
print("Plot 2 saved!")

# --- Plot 3: Total Mass vs Dark Matter Fraction ---
fig, ax = plt.subplots(figsize=(6, 5))

for name, m_tot, frac, color in zip(names,
                                      [M_tot_3198, M_tot_6503, M_tot_ddo],
                                      [frac_3198, frac_6503, frac_ddo],
                                      colors):
    ax.scatter(np.log10(m_tot), frac * 100,
               s=150, label=name, color=color,
               edgecolors='white', linewidth=1.5, zorder=3)

ax.set_xlabel('log₁₀(Total Mass / M☉)')
ax.set_ylabel('Dark Matter Fraction (%)')
ax.set_title('Total Mass vs. Dark Matter Fraction', fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(50, 100)
plt.tight_layout()
plt.savefig('figures/dm_fraction_scatter.png', dpi=150)
plt.show()
print("Plot 3 saved!")