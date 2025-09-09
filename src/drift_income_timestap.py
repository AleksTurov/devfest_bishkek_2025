import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from river import drift as river_drift
from sklearn.datasets import load_breast_cancer

# Краткое пояснение (для слайда/репозитория):
# ADWIN — онлайн-детектор дрейфа. Метод поддерживает адаптивное окно
# последних наблюдений и возвращает True из update(value), когда
# статистика окна меняется существенно (change detected).
# Параметр delta контролирует чувствительность: меньшее delta -> более чувствительный.

# 1) Берём примерный датасет и один признак
data = load_breast_cancer(as_frame=True)
df = data.frame[["mean radius"]].rename(columns={"mean radius": "value"}).reset_index(drop=True)

# 2) Даты и искусственный дрейф во второй половине — для наглядности
n = len(df)
df["date"] = pd.date_range("2024-01-01", periods=n, freq="D")
mid = n // 2
df.loc[mid:, "value"] = df.loc[mid:, "value"] + 8.0 + np.random.normal(0, 0.5, n - mid)  # инжект дрейфа

# 3) Онлайн-детектор ADWIN: итерируемся по потоку и отмечаем моменты изменений
#    detector.update(x) -> True означает, что ADWIN считает, что распределение изменилось.
detector = river_drift.ADWIN(delta=1e-6)  # уменьшите delta для большей чувствительности
alerts = [bool(detector.update(float(x))) for x in df["value"].to_numpy()]
df["alert"] = alerts

# 4) Простой порог как fallback / пояснение (не обязателен для demo)
#    Здесь показываем альтернативный подход: сдвиг скользящего среднего относительно baseline.
baseline = df["value"].iloc[:30].mean()
baseline_std = df["value"].iloc[:30].std(ddof=0)
simple_alert = (df["value"].rolling(7, center=True, min_periods=1).mean() - baseline).abs() > 3.0 * (baseline_std + 1e-12)

# 5) Визуализация для слайда: ряд + сглаживание + вертикальные линии и маркеры алертов
out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "docs"))
os.makedirs(out_dir, exist_ok=True)
out_png = os.path.join(out_dir, "drift_presentation_explained.png")

plt.figure(figsize=(12, 4), dpi=120)
# История наблюдений (бледно)
plt.plot(df["date"], df["value"], color="lightgray", label="value (history)")
# Сглаживание для визуализации тренда
plt.plot(df["date"],
         df["value"].rolling(7, center=True, min_periods=1).mean(),
         color="tab:green", linestyle="--", label="rolling mean (7)")
# Вертикальные линии — места, где ADWIN вернул True
for d, a in zip(df["date"], df["alert"]):
    if a:
        plt.axvline(d, color="red", alpha=0.7, linewidth=1.6)  # ADWIN detected change here
# Маркеры — сами точки срабатывания
if df["alert"].any():
    plt.scatter(df.loc[df["alert"], "date"], df.loc[df["alert"], "value"],
                color="red", zorder=5, label="ADWIN alert")
# Для сравнения можно показать простой порогный алерт (опционально)
if simple_alert.any():
    plt.scatter(df.loc[simple_alert, "date"], df.loc[simple_alert, "value"],
                color="orange", zorder=4, marker="x", label="simple threshold")

plt.title("Detected Data Drift — ADWIN (краткое объяснение в коде)")
plt.xlabel("Дата")
plt.ylabel("Значение признака")
plt.legend(loc="upper left")
plt.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(out_png, bbox_inches="tight")
plt.close()