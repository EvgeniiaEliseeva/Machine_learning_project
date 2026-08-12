"""
In this file, I will be preparing data for modeling, training and evaluation of the three classifiers (logistic regression, decision tree and ANN) on the Credit Risk dataset.

I will be showing the confusion matrices, ROC curves and a metric comparison table for the three mention models.
"""

# Подключаем библиотеки для работы с данными и визуализациями
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Импортируем инструменты для разделения данных на обучающую и тестовую выборки
from sklearn.model_selection import train_test_split

# Импортируем StandardScaler для стандартизации числовых признаков
from sklearn.preprocessing import StandardScaler

# Импортируем модель логистической регрессии
from sklearn.linear_model import LogisticRegression

# Импортируем дерево решений и функцию для вывода его правил
from sklearn.tree import DecisionTreeClassifier, export_text

# Импортируем модель нейронной сети
from sklearn.neural_network import MLPClassifier

# Импортируем основные метрики для оценки качества моделей
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve)

# Путь к исходному набору данных
DATA = "credit_risk_dataset.csv"

# Фиксируем случайное состояние для воспроизводимости результатов
RANDOM_STATE = 42

# Основная цветовая палитра для визуализаций
PAL = ["#4C72B0", "#DD8452"]

# Устанавливаем качество отображения графиков
plt.rcParams["figure.dpi"] = 110

# Применяем стиль с белым фоном и сеткой
sns.set_style("whitegrid")


def load_and_prepare():
    # Clean, encode and split. Imputation/scaling are fitted on TRAIN only.
    # Загружаем исходный набор данных
    df = pd.read_csv(DATA)

    # Сохраняем первоначальное количество строк
    n0 = len(df)

    # Удаляем полностью дублирующиеся строки
    df = df.drop_duplicates()

    # Удаляем нереалистичные значения возраста и стажа работы
    df = df[(df["person_age"] <= 100) &
            ((df["person_emp_length"] <= 60) | df["person_emp_length"].isna())]
    # Выводим количество строк до и после очистки
    print(f"Rows: {n0} -> {len(df)} (removed {n0 - len(df)})")

    # Преобразуем кредитные рейтинги A-G в числовые значения 0-6
    df["loan_grade"] = df["loan_grade"].map({g: i for i, g in enumerate("ABCDEFG")})
    
    # Преобразуем информацию о предыдущем дефолте: Y становится 1, остальные значения становятся 0
    df["cb_person_default_on_file"] = (df["cb_person_default_on_file"] == "Y").astype(int)

    # Преобразуем категориальные признаки в числовые с помощью one-hot encoding
    df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], drop_first=True)

    # Сохраняем целевую переменную отдельно
    y = df["loan_status"]
        
    # Все остальные признаки используем как входные данные модели
    X = df.drop(columns="loan_status")
    # Разделяем данные на обучающую и тестовую выборки.
    # 80% данных используются для обучения, 20% — для тестирования.
    # stratify=y сохраняет одинаковое соотношение классов в обеих выборках.
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    # Рассчитываем медианную процентную ставку для каждого кредитного рейтинга только на обучающей выборке
    grade_med = Xtr.groupby("loan_grade")["loan_int_rate"].median()

    # Рассчитываем общую медианную процентную ставку
    glob_rate = Xtr["loan_int_rate"].median()

    # Рассчитываем медианное значение стажа работы
    emp_med = Xtr["person_emp_length"].median()

    # Обрабатываем пропущенные значения отдельно в обучающей и тестовой выборках
    for part in (Xtr, Xte):
        # Сначала заполняем пропущенную процентную ставку медианой соответствующего кредитного рейтинга.
        # Если значение всё ещё отсутствует, используем общую медиану.
        part["loan_int_rate"] = (part["loan_int_rate"]
                                 .fillna(part["loan_grade"].map(grade_med))
                                 .fillna(glob_rate))

        # Заполняем пропущенный стаж работы медианным значением
        part["person_emp_length"] = part["person_emp_length"].fillna(emp_med)

    # Создаём объект для стандартизации данных и обучаем его только на обучающей выборке
    scaler = StandardScaler().fit(Xtr)

    # Стандартизируем обучающую и тестовую выборки
    Xtr_s, Xte_s = scaler.transform(Xtr), scaler.transform(Xte)

    # Возвращаем подготовленные данные
    return X.columns, Xtr, Xte, ytr, yte, Xtr_s, Xte_s


def train_models(Xtr, ytr, Xtr_s):
    # Обучаем логистическую регрессию на стандартизированных данных
    logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_STATE).fit(Xtr_s, ytr)

    # Обучаем дерево решений с максимальной глубиной 6
    tree = DecisionTreeClassifier(criterion="entropy", max_depth=6, random_state=RANDOM_STATE).fit(Xtr, ytr)

    # Обучаем нейронную сеть с двумя скрытыми слоями: первый слой содержит 32 нейрона, второй — 16
    ann = MLPClassifier(hidden_layer_sizes=(32, 16), activation="relu", max_iter=400,
                        early_stopping=True, random_state=RANDOM_STATE).fit(Xtr_s, ytr)
    
    # Возвращаем три обученные модели
    return logreg, tree, ann


def evaluate(models, yte):
    # Создаём структуры для хранения результатов, confusion matrices и ROC-кривых
    rows, cms, rocs = [], {}, {}

    # Проходим по каждой модели
    for name, (m, Xt) in models.items():

        # Получаем предсказанный класс: 0 или 1
        pred = m.predict(Xt)

        # Получаем вероятность принадлежности к классу дефолта
        proba = m.predict_proba(Xt)[:, 1]

        # Рассчитываем основные метрики и сохраняем их в общий список
        rows.append([name, accuracy_score(yte, pred), precision_score(yte, pred),
                     recall_score(yte, pred), f1_score(yte, pred), roc_auc_score(yte, proba)])

        # Сохраняем confusion matrix для каждой модели
        cms[name] = confusion_matrix(yte, pred)

        # Сохраняем данные для построения ROC-кривой
        rocs[name] = roc_curve(yte, proba)


    # Создаём таблицу со всеми метриками
    res = pd.DataFrame(
        rows, columns=["Model", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    ).set_index("Model")

    # Возвращаем результаты оценки
    return res, cms, rocs


def plot_confusion(cms):
    # Создаём три графика рядом - по одному для каждой модели
    fig, ax = plt.subplots(1, 3, figsize=(14, 4))

    # Проходим по confusion matrix каждой модели
    for i, (name, cm) in enumerate(cms.items()):
        # Строим тепловую карту confusion matrix
        sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=ax[i], cbar=False,
                    xticklabels=["Good", "Default"], yticklabels=["Good", "Default"])
        
        # Добавляем название модели и подписи осей
        ax[i].set_title(name); ax[i].set_xlabel("Predicted"); ax[i].set_ylabel("Actual")
        
    # Добавляем общий заголовок
    plt.suptitle("Confusion matrices (test set)", fontweight="bold")

    # Выравниваем расположение графиков
    plt.tight_layout(); 
    
    # Сохраняем изображение
    plt.savefig("model_confusion.png", bbox_inches="tight"); 
    
    # Закрываем график
    plt.close()


def plot_roc(rocs, res):
    # Создаём область для ROC-кривых
    plt.figure(figsize=(6, 5.5))

    # Строим ROC-кривую для каждой модели
    for name, (fpr, tpr, _) in rocs.items():

        # Добавляем кривую и значение ROC-AUC в легенду
        plt.plot(fpr, tpr, label=f"{name} (AUC={res.loc[name, 'ROC-AUC']:.3f})")

    # Добавляем диагональную линию случайного классификатора
    plt.plot([0, 1], [0, 1], "k--", lw=1)

    # Подписываем оси графика
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")

    # Добавляем название и легенду
    plt.title("ROC curves"); plt.legend()

    # Выравниваем элементы
    plt.tight_layout()

    # Сохраняем ROC-кривые
    plt.savefig("model_roc.png", bbox_inches="tight")

    # Закрываем график
    plt.close()

def main():
    # Загружаем и подготавливаем данные
    cols, Xtr, Xte, ytr, yte, Xtr_s, Xte_s = load_and_prepare()

    # Обучаем три модели
    logreg, tree, ann = train_models(Xtr, ytr, Xtr_s)

    # Связываем каждую модель с подходящей тестовой выборкой.
    # Logistic Regression и ANN используют стандартизированные данные,
    # а Decision Tree использует исходный масштаб признаков.
    models = {
        "Logistic Regression": (logreg, Xte_s),
        "Decision Tree": (tree, Xte),
        "ANN (MLP)": (ann, Xte_s),
    }

    # Оцениваем качество всех моделей
    res, cms, rocs = evaluate(models, yte)

    # Выводим результаты моделей на тестовой выборке
    print("\n=== RESULTS (test set) ===")

    # Округляем значения до четырёх знаков после запятой
    print(res.round(4).to_string())

    # Строим confusion matrices
    plot_confusion(cms)

    # Строим ROC-кривые
    plot_roc(rocs, res)

    # Сохраняем таблицу с метриками в CSV-файл
    res.round(4).to_csv("results.csv")


    # Получаем важность признаков из дерева решений и сортируем их от наиболее важных к менее важным
    fi = pd.Series(tree.feature_importances_, index=cols).sort_values(ascending=False)
    # Выводим наиболее важные признаки
    print("\nTop tree feature importances:")

    # Показываем первые 8 признаков
    print(fi.head(8).round(3).to_string())

    # Выводим более простые и читаемые правила дерева решений
    print("\nReadable decision rules (depth 3):")

    # Создаём отдельное небольшое дерево глубиной 3 для более простого объяснения его решений
    shallow = DecisionTreeClassifier(max_depth=3, random_state=RANDOM_STATE).fit(Xtr, ytr)

    # Выводим правила дерева в текстовом виде
    print(export_text(shallow, feature_names=list(cols), max_depth=3))

    # Выводим список созданных файлов
    print("Saved: model_confusion.png, model_roc.png, results.csv")


# Запускаем функцию main(), если файл был запущен напрямую
if __name__ == "__main__":
    main()
