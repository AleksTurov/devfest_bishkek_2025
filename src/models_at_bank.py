import matplotlib.pyplot as plt
import os

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_axis_off()

pd_pos = (0.1, 0.6)
lgd_pos = (0.1, 0.3)
ead_pos = (0.1, 0.0)
el_pos = (0.7, 0.45)

def draw_box(pos, text):
    ax.add_patch(plt.Rectangle((pos[0], pos[1]), 0.25, 0.18, fill=True, facecolor="#e6f2ff", edgecolor="black", lw=1.5))
    ax.text(pos[0]+0.125, pos[1]+0.09, text, ha="center", va="center", fontsize=12, weight="bold")

draw_box(pd_pos, "PD\n(Probability of Default)")
draw_box(lgd_pos, "LGD\n(Loss Given Default)")
draw_box(ead_pos, "EAD\n(Exposure at Default)")
draw_box(el_pos, "EL\n(Expected Loss)")

arrowprops = dict(arrowstyle="->", lw=2, color="black")
ax.annotate("", xy=(el_pos[0], el_pos[1]+0.09), xytext=(pd_pos[0]+0.25, pd_pos[1]+0.09), arrowprops=arrowprops)
ax.annotate("", xy=(el_pos[0], el_pos[1]+0.09), xytext=(lgd_pos[0]+0.25, lgd_pos[1]+0.09), arrowprops=arrowprops)
ax.annotate("", xy=(el_pos[0], el_pos[1]+0.01), xytext=(ead_pos[0]+0.25, ead_pos[1]+0.09), arrowprops=arrowprops)

ax.text(0.7, 0.25, "EL = PD × LGD × EAD", fontsize=14, ha="center", weight="bold")

out_path = "/workspaces/devfest_bishkek_2025/docs/pd_lgd_ead_el_diagram.png"

os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
plt.savefig(out_path, bbox_inches="tight")
plt.close()
print(f"Saved: {os.path.abspath(out_path)}")
