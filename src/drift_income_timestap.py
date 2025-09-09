import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.dates import date2num
from river import drift
from sklearn.datasets import load_breast_cancer

# Берём готовый датасет и один признак
data = load_breast_cancer(as_frame=True)
df = data.frame[[ "mean radius" ]].rename(columns={"mean radius": "value"}).reset_index(drop=True)

# Даты и инжект дрейфа во второй половине
n = len(df)
df["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
mid = n // 2
df.loc[mid:, "value"] = df.loc[mid:, "value"] + 5.0 + np.random.normal(0, 0.5, n - mid)

# Онлайн-детектор ADWIN (stream)
detector = drift.ADWIN(delta=0.002)
alerts = []
for v in df["value"]:
    alerts.append(bool(detector.update(float(v))))
df["alert"] = alerts

# Сохранить лог алертов
out_csv = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "stream_alerts_simple.csv"))
os.makedirs(os.path.dirname(out_csv), exist_ok=True)
df.loc[df["alert"], ["date", "value"]].to_csv(out_csv, index=False)

# Статичный график (простая визуализация)
out_png = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "stream_simple.png"))
plt.figure(figsize=(10,4), dpi=120)
plt.plot(df["date"], df["value"], color="tab:blue", label="value")
if df["alert"].any():
    plt.scatter(df.loc[df["alert"], "date"], df.loc[df["alert"], "value"], color="red", marker="v", zorder=5, label="alert")
plt.title("Stream (simple) — ADWIN alerts")
plt.xlabel("Дата")
plt.ylabel("value")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(out_png, bbox_inches="tight")
plt.close()
print(f"Saved static plot: {out_png}")
print(f"Saved alerts CSV: {out_csv}")

# Простая анимация: точка движется по времени, алерты подсвечиваются
dates_num = date2num(df["date"])
vals = df["value"].to_numpy()
alerts_arr = df["alert"].to_numpy()

fig, ax = plt.subplots(figsize=(10,4), dpi=120)
ax.plot_date(dates_num, vals, color="lightgray", linewidth=1)  # история
ax.set_xlim(dates_num[0], dates_num[-1])
ax.set_ylim(vals.min() - 2, vals.max() + 2)
ax.set_title("Streaming (animation) — ADWIN alerts")
ax.set_xlabel("Дата")

current_scatter = ax.scatter([], [], s=80, color="tab:blue", zorder=5)
alert_scatter = ax.scatter([], [], s=80, color="red", marker='v', zorder=6)
time_line = ax.axvline(dates_num[0], color="gray", linestyle=":", alpha=0.6)

def init():
    current_scatter.set_offsets(np.empty((0,2)))
    alert_scatter.set_offsets(np.empty((0,2)))
    time_line.set_xdata([dates_num[0], dates_num[0]]) 
    return current_scatter, alert_scatter, time_line

def update(i):
    cur_x = dates_num[i]
    cur_y = vals[i]
    current_scatter.set_offsets([[cur_x, cur_y]])
    mask = alerts_arr[:i+1]
    if mask.any():
        ad = dates_num[:i+1][mask]
        av = vals[:i+1][mask]
        alert_scatter.set_offsets(np.column_stack([ad, av]))
    else:
        alert_scatter.set_offsets(np.empty((0,2)))
    time_line.set_xdata([cur_x, cur_x]) 
    current_scatter.set_color('red' if alerts_arr[i] else 'tab:blue')
    return current_scatter, alert_scatter, time_line

anim = FuncAnimation(fig, update, frames=len(df), init_func=init, interval=120, blit=False)

out_gif = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs", "stream_simple.gif"))
writer = PillowWriter(fps=8)
anim.save(out_gif, writer=writer)
print(f"Saved animation: {out_gif}")
