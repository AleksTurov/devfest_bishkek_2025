import numpy as np
import os
import matplotlib.pyplot as plt


def _roc_curve_manual(y_true, y_score):
    """Вычислить ROC-кривую вручную.

    Возвращает fpr, tpr, thresholds аналогично sklearn.metrics. Все массивы numpy.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Проверки входных данных
    if y_true.shape[0] != y_score.shape[0]:
        raise ValueError('y_true и y_score должны иметь одинаковую длину')

    # Общее число положительных и отрицательных примеров
    P = np.sum(y_true == 1)
    N = np.sum(y_true == 0)
    if P == 0 or N == 0:
        raise ValueError('Требуются как положительные, так и отрицательные примеры (P>0 и N>0)')

    # Сортируем по убыванию оценок
    desc_idx = np.argsort(-y_score, kind='mergesort')
    y_true_sorted = y_true[desc_idx]
    y_score_sorted = y_score[desc_idx]

    # Кумулятивные суммы True Positive и False Positive
    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    # Индексы, где значение score меняется — точки для отрисовки
    distinct_idxs = np.where(np.diff(y_score_sorted) != 0)[0]
    threshold_idxs = np.r_[distinct_idxs, y_true_sorted.size - 1]

    tpr = tps[threshold_idxs] / float(P)
    fpr = fps[threshold_idxs] / float(N)
    thresholds = y_score_sorted[threshold_idxs]

    # Добавляем начальные и конечные точки (0,0) и (1,1)
    fpr = np.r_[0.0, fpr, 1.0]
    tpr = np.r_[0.0, tpr, 1.0]
    thresholds = np.r_[thresholds, 0.0]

    return fpr, tpr, thresholds


def _auc_manual(fpr, tpr):
    """Вычислить площадь под ROC-кривой методом трапеций.

    Принимает отсортированные массивы fpr и tpr.
    """
    # Используем numpy.trapezoid (вместо устаревающего trapz)
    return np.trapezoid(tpr, fpr)


def calculate_auc_roc(y_true, y_pred):
    """Вычислить AUC-ROC без использования внешних библиотек metrics.

    Args:
        y_true (list или np.array): истинные бинарные метки (0 или 1)
        y_pred (list или np.array): оценки/вероятности положительного класса

    Returns:
        float: AUC-ROC
    """
    fpr, tpr, _ = _roc_curve_manual(y_true, y_pred)
    return _auc_manual(fpr, tpr)


def plot_roc_curve(y_true, y_pred, out_path=None):
    """Построить и сохранить ROC-кривую.

    Аргументы:
        y_true, y_pred -- как в calculate_auc_roc
        out_path -- путь для сохранения PNG; по умолчанию сохраняет в ../docs/roc_curve.png
    """
    fpr, tpr, _ = _roc_curve_manual(y_true, y_pred)
    auc = _auc_manual(fpr, tpr)

    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.3f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.5)')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)

    # Определяем путь сохранения относительно текущего файла
    if out_path is None:
        here = os.path.dirname(os.path.abspath(__file__))
        out_path = os.path.abspath(os.path.join(here, '..', 'docs', 'roc_curve.png'))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path)
    plt.close()


if __name__ == "__main__":
    # Пример данных
    y_pred = [0.55, 0.8, 0.3, 0.95, 0.7]
    y_true = [0, 1, 1, 1, 0]

    # Вычисляем AUC вручную
    auc = calculate_auc_roc(y_true, y_pred)
    print(f"AUC-ROC score: {auc:.4f}")

    # Рисуем и сохраняем ROC-кривую
    plot_roc_curve(y_true, y_pred)
    print("ROC curve saved as 'docs/roc_curve.png'")
    
