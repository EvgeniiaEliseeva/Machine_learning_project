"""
In this file, I will be trying to understand the data and perform some exploratory data analysis (EDA) 
to get a better understanding of the dataset.
I will configure few pictures to visualize the data and understand the relationships:

    eda_1_target.png       - target class balance
    eda_2_numeric.png      - numeric feature distributions
    eda_3_categorical.png  - default rate by categorical feature
    eda_4_drivers.png      - strongest numeric drivers + correlation heatmap

"""
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Путь к исходному набору данных
DATA = "credit_risk_dataset.csv"

# Основная цветовая палитра для визуализаций
PAL = ["#4C72B0", "#DD8452"]

# Устанавливаем плотность отображения (DPI) для всех создаваемых графиков.
# Это обеспечивает более чёткое отображение текста, линий и других элементов.
plt.rcParams["figure.dpi"] = 110

# Применяем стиль "whitegrid", который добавляет светлый фон с координатной сеткой.
# Такой стиль облегчает визуальное сравнение значений и делает графики более читаемыми при анализе данных.
sns.set_style("whitegrid")


def main():
    # Загружаем исходный набор данных из CSV-файла в объект DataFrame
    df = pd.read_csv(DATA)

    # Выводим количество строк и столбцов в наборе данных
    print("Shape:", df.shape)

    # Подсчитываем количество пропущенных значений в каждом столбце
    missing_values = df.isna().sum()

    # Выводим только столбцы, содержащие хотя бы одно пропущенное значение
    print("\nMissing values:")
    print(missing_values[missing_values > 0])

    # Метод value_counts() подсчитывает количество каждого уникального значения
    # в столбце. Параметр normalize=True возвращает не количество, а долю,
    # которая затем переводится в проценты и выводится на экран.
    print("\nTarget balance (%):")
    print((df["loan_status"].value_counts(normalize=True) * 100).round(2))

    # Подсчитываем количество полностью дублирующихся строк
    print("Duplicate rows:", df.duplicated().sum())

    # Создаём область для двух графиков, расположенных рядом
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    # Подсчитываем количество хороших кредитов и дефолтов и сортируем их по значениям 0 и 1
    vc = df["loan_status"].value_counts().sort_index()

    # Создаём столбчатую диаграмму количества заявителей для хороших кредитов и дефолтов
    ax[0].bar(["Good (0)", "Default (1)"], vc.values, color=PAL)

    # Добавляем точное количество заявителей над каждым столбцом
    for i, v in enumerate(vc.values):
        ax[0].text(i, v + 300, f"{v:,}", ha="center", fontweight="bold")

    # Добавляем название графика и подпись вертикальной оси
    ax[0].set_title("Target counts"); ax[0].set_ylabel("applicants")

    # Создаём круговую диаграмму для отображения соотношения классов width=0.45 делает диаграмму в форме кольца
    ax[1].pie(vc.values, labels=["Good 78.2%", "Default 21.8%"], colors=PAL,
              startangle=90, wedgeprops=dict(width=0.45))

    # Добавляем название круговой диаграммы
    ax[1].set_title("Class balance")

    # Добавляем общий заголовок для двух графиков
    plt.suptitle("Target: loan_status", fontweight="bold")

    # Автоматически выравниваем расположение элементов графика
    plt.tight_layout()

    # Сохраняем полученный график в PNG-файл
    plt.savefig("eda_1_target.png", bbox_inches="tight")

    # Закрываем график после сохранения
    plt.close()

    # Создаём список числовых признаков для дальнейшего анализа
    num = ["person_age", "person_income", "person_emp_length", "loan_amnt",
           "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length"]

    # Создаём область из 8 графиков: 2 строки и 4 столбца
    fig, axes = plt.subplots(2, 4, figsize=(15, 7))

    # Проходим по каждому числовому признаку
    for i, c in enumerate(num):

        # Удаляем пропущенные значения перед построением графика
        d = df[c].dropna()

        # Считаем проценты пропуценных чисел
        sum_of_na = df[c].isna().sum()
        if sum_of_na != 0:
            print(f'For {c}, percentage of missing values is {(sum_of_na/len(df))*100}')

        # Для дохода используем логарифмическое преобразование, так как значения дохода имеют большой разброс
        if c == "person_income":
            axes.flat[i].hist(np.log10(d), bins=40, color=PAL[0])
            axes.flat[i].set_xlabel("log10(income)")

        # Для остальных признаков строим обычную гистограмму
        else:
            axes.flat[i].hist(d, bins=40, color=PAL[0])

        # Используем название признака как заголовок графика
        axes.flat[i].set_title(c, fontsize=9)

    # Последняя область остаётся пустой, поэтому отключаем её
    axes.flat[-1].axis("off")

    # Добавляем общий заголовок для распределений числовых признаков
    plt.suptitle("Numeric feature distributions "
                 "(note age & emp_length tails = data errors)", fontweight="bold")

    # Выравниваем расположение графиков
    plt.tight_layout()

    # Сохраняем график распределения числовых признаков
    plt.savefig("eda_2_numeric.png", bbox_inches="tight")

    # Закрываем график
    plt.close()
    
    # Создаём список категориальных признаков для анализа
    cats = ["person_home_ownership", "loan_intent", "loan_grade", "cb_person_default_on_file"]

    # Создаём четыре графика в формате 2 на 2
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Проходим по каждому категориальному признаку
    for i, c in enumerate(cats):

        # Группируем данные по категориям и рассчитываем процент дефолтов для каждой категории
        g = df.groupby(c)["loan_status"].mean().sort_values(ascending=False) * 100

        # Строим столбчатую диаграмму с процентом дефолтов
        axes.flat[i].bar(g.index, g.values, color=PAL[1])

        # Добавляем процент дефолтов над каждым столбцом
        for j, v in enumerate(g.values):
            axes.flat[i].text(j, v + 0.5, f"{v:.0f}%", ha="center", fontsize=8)

        # Добавляем горизонтальную линию, которая показывает общий процент дефолтов в датасете
        axes.flat[i].axhline(21.82, color="gray", ls="--", lw=1, label="overall 21.8%")

        # Добавляем название графика
        axes.flat[i].set_title(f"Default rate by {c}", fontsize=10)

        # Подписываем вертикальную ось
        axes.flat[i].set_ylabel("% default")

        # Добавляем легенду для линии общего уровня дефолтов
        axes.flat[i].legend(fontsize=7)

        # Поворачиваем подписи категорий для лучшей читаемости
        axes.flat[i].tick_params(axis="x", rotation=30)

    # Добавляем общий заголовок
    plt.suptitle("How default rate varies across categories", fontweight="bold")

    # Выравниваем расположение графиков
    plt.tight_layout()

    # Сохраняем графики категориальных признаков
    plt.savefig("eda_3_categorical.png", bbox_inches="tight")

    # Закрываем график
    plt.close()


    # Создаём три графика для анализа наиболее сильных числовых связей с дефолтом
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # Анализируем отношение суммы кредита к доходу и процентную ставку
    for ax, col in [(axes[0], "loan_percent_income"),(axes[1], "loan_int_rate")]:

        # Удаляем строки с пропущенными значениями и создаём отдельную копию данных
        tmp = df.dropna(subset=[col]).copy()

        # Разделяем значения признака на 6 примерно равных групп
        tmp["bin"] = pd.qcut(tmp[col], 6, duplicates="drop")

        # Рассчитываем процент дефолтов для каждой группы
        g = tmp.groupby("bin", observed=True)["loan_status"].mean() * 100

        # Строим линейный график изменения уровня дефолтов
        ax.plot(range(len(g)), g.values, "o-", color=PAL[0])

        # Устанавливаем позиции подписей по горизонтальной оси
        ax.set_xticks(range(len(g)))

        # В качестве подписей используем среднее значение каждого интервала
        ax.set_xticklabels([f"{iv.mid:.2f}" for iv in g.index], rotation=30, fontsize=8)

        # Добавляем название графика и подпись вертикальной оси
        ax.set_title(f"Default rate vs {col}", fontsize=10); ax.set_ylabel("% default")


    # Рассчитываем корреляцию между числовыми признаками и целевой переменной loan_status
    corr = df[num + ["loan_status"]].corr()

    # Строим тепловую карту корреляций.
    # Значения внутри ячеек показывают силу и направление связи между различными признаками.
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=axes[2],
                cbar=False, annot_kws={"size": 7})

    # Добавляем название тепловой карты
    axes[2].set_title("Correlation (numeric + target)", fontsize=10)

    # Добавляем общий заголовок
    plt.suptitle("Strongest numeric relationships with default", fontweight="bold")

    # Выравниваем расположение всех элементов
    plt.tight_layout()

    # Сохраняем итоговый график
    plt.savefig("eda_4_drivers.png", bbox_inches="tight")

    # Закрываем график
    plt.close()


    # Выводим сообщение с названиями всех созданных изображений
    print("\nSaved: eda_1_target.png, eda_2_numeric.png, eda_3_categorical.png, eda_4_drivers.png")


# Запускаем функцию main(), если файл был запущен напрямую
if __name__ == "__main__":
    main()
