"""
03_generate_report.py  -  Build the final project report PDF
Loan Default Prediction project

Assembles the written report and embeds the figures produced by
01_eda.py and 02_modeling.py. The metrics table is read from results.csv.

Prerequisites (run these first):
    python 01_eda.py
    python 02_modeling.py

Run:  python 03_generate_report.py
Output: Loan_Default_Prediction_Report.pdf

"""


# Указываем автора проекта
AUTHORS = [
    "Evgeniia Eliseeva  (ID: 89231137)",
]

# Указываем дату отчёта
DATE = "July 2026"


# Подключаем pandas для работы с таблицей результатов
import pandas as pd

# Импортируем PIL для получения размеров изображений
from PIL import Image as PILImage

# Импортируем формат страницы A4
from reportlab.lib.pagesizes import A4

# Импортируем миллиметры для задания размеров и отступов
from reportlab.lib.units import mm

# Импортируем цвета для оформления PDF
from reportlab.lib import colors

# Импортируем инструменты для создания текстовых стилей
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# Импортируем варианты выравнивания текста
from reportlab.lib.enums import TA_JUSTIFY, TA_CENTER

# Импортируем основные элементы для создания PDF-документа
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Image, Table,
                                TableStyle, HRFlowable)


# Получаем ширину и высоту страницы A4
PAGE_W, PAGE_H = A4

# Устанавливаем левый и правый отступ по 20 мм
LMARGIN = RMARGIN = 20 * mm

# Рассчитываем доступную ширину страницы
USABLE = PAGE_W - LMARGIN - RMARGIN


# Создаём основные цвета для оформления отчёта
NAVY = colors.HexColor("#1F3A5F")
ACCENT = colors.HexColor("#4C72B0")
GREY = colors.HexColor("#555555")


# Загружаем стандартный набор стилей ReportLab
ss = getSampleStyleSheet()

# Создаём собственные стили для разных частей отчёта
styles = {

    # Стиль для главного заголовка
    "title": ParagraphStyle("title", parent=ss["Title"], fontName="Helvetica-Bold",
                            fontSize=20, textColor=NAVY, spaceAfter=6, leading=24),

    # Стиль для подзаголовка
    "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=12,
                               textColor=GREY, alignment=TA_CENTER, spaceAfter=18, leading=16),

    # Стиль для имени автора и даты
    "author": ParagraphStyle("author", fontName="Helvetica", fontSize=11,
                             alignment=TA_CENTER, leading=16),

    # Стиль для основных разделов
    "h1": ParagraphStyle("h1", fontName="Helvetica-Bold", fontSize=14, textColor=NAVY,
                         spaceBefore=16, spaceAfter=6, leading=18),

    # Стиль для подразделов
    "h2": ParagraphStyle("h2", fontName="Helvetica-Bold", fontSize=11.5, textColor=ACCENT,
                         spaceBefore=10, spaceAfter=4, leading=15),

    # Основной стиль текста
    "body": ParagraphStyle("body", fontName="Helvetica", fontSize=10, alignment=TA_JUSTIFY,
                           leading=15, spaceAfter=6, textColor=colors.HexColor("#1a1a1a")),

    # Стиль для подписей к таблицам и изображениям
    "cap": ParagraphStyle("cap", fontName="Helvetica-Oblique", fontSize=8.5, textColor=GREY,
                          alignment=TA_CENTER, spaceBefore=3, spaceAfter=10, leading=11),

    # Стиль для текста Abstract
    "abstract": ParagraphStyle("abstract", fontName="Helvetica", fontSize=9.5, alignment=TA_JUSTIFY,
                               leading=14, leftIndent=10, rightIndent=10),

    # Стиль для списка источников
    "ref": ParagraphStyle("ref", fontName="Helvetica", fontSize=9, leading=14, spaceAfter=4,
                          leftIndent=14, firstLineIndent=-14),
}


# Функция для добавления изображения в отчёт
def img(path, width=USABLE):

    # Получаем исходную ширину и высоту изображения
    iw, ih = PILImage.open(path).size

    # Масштабируем изображение, сохраняя его пропорции
    return Image(path, width=width, height=width * ih / iw)


# Функция создаёт таблицу с метриками моделей
def results_table():

    # Build the metrics table from results.csv, bolding the best value per metric.

    # Загружаем результаты моделей из CSV-файла
    df = pd.read_csv("results.csv", index_col=0)

    # Создаём заголовок таблицы
    header = ["Model"] + list(df.columns)

    # Начинаем таблицу со строки заголовков
    data = [header]

    # Проходим по каждой модели в таблице результатов
    for name, row in df.iterrows():

        # Добавляем название модели и значения метрик, округлённые до трёх знаков
        data.append([name] + [f"{v:.3f}" for v in row])

    # Создаём таблицу и задаём ширину столбцов
    t = Table(data, colWidths=[USABLE * 0.32] + [USABLE * 0.136] * (len(header) - 1))

    # Настраиваем внешний вид таблицы
    style = [
        # Цвет фона заголовка таблицы
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        # Белый цвет текста заголовка
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        # Жирный шрифт в заголовке
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        # Размер шрифта всей таблицы
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        # Обычный шрифт для названий моделей
        ("FONTNAME", (0, 1), (0, -1), "Helvetica"),
        # Выравниваем значения метрик по центру
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        # Чередуем цвет строк для удобства чтения
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]),
        # Добавляем границы ячеек
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        # Добавляем вертикальные отступы внутри ячеек
        ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
        # Добавляем левый отступ для первого столбца
        ("LEFTPADDING", (0, 0), (0, -1), 6),
    ]

    # Проходим по каждой метрике
    for j, col in enumerate(df.columns, start=1):

        # Находим строку с лучшим значением этой метрики
        best_row = df[col].values.argmax() + 1

        # Выделяем лучшее значение жирным шрифтом
        style.append(("FONTNAME", (j, best_row), (j, best_row), "Helvetica-Bold"))

        # Выделяем лучшее значение другим цветом
        style.append(("TEXTCOLOR", (j, best_row), (j, best_row), colors.HexColor("#B8860B")))

    # Применяем все стили к таблице
    t.setStyle(TableStyle(style))

    # Возвращаем готовую таблицу
    return t


# Основная функция для создания PDF-отчёта
def build():

    # Создаём список всех элементов будущего PDF
    S = []

    # Создаём короткую функцию для добавления текста в документ
    P = lambda t, s="body": S.append(Paragraph(t, styles[s]))

    # Добавляем небольшой отступ сверху
    S.append(Spacer(1, 10))

    # Добавляем название проекта
    P("Loan Default Prediction Using Classification Methods", "title")

    # Добавляем подзаголовок
    P("A credit-risk binary-classification study following the CRISP-DM methodology", "subtitle")

    # Добавляем декоративную горизонтальную линию
    S.append(HRFlowable(width="40%", thickness=1.2, color=ACCENT,
                        spaceBefore=2, spaceAfter=14, hAlign="CENTER"))
    
    # Добавляем имя каждого автора
    for a in AUTHORS:
        P(a, "author")

    # Добавляем дату
    P(DATE, "author")

    # Добавляем отступ после информации об авторе
    S.append(Spacer(1, 16))

    # Добавляем заголовок Abstract
    P("Abstract", "h1")

    # Добавляем текст Abstract
    P("In this project, I explored whether loan applicants' information can be used to predict if they are likely "
      "to repay their loan or default. I used the Kaggle Credit Risk Dataset, which contains information about "
      "32,581 loan applications. Following the CRISP-DM process, I first explored and prepared the data before "
      "training three different classification models: Logistic Regression, a Decision Tree and an Artificial "
      "Neural Network. I compared their performance using several measures, including accuracy, precision, recall, "
      "F1-score and ROC-AUC. The Decision Tree provided the best overall balance between performance and "
      "interpretability, with an accuracy of around 92%. The neural network was better at identifying applicants "
      "who might default, achieving the highest recall and ROC-AUC. The analysis showed that loan grade, the "
      "loan-to-income ratio and home-ownership status were the most important factors related to default risk.", "abstract")

    # Добавляем небольшой отступ
    S.append(Spacer(1, 4))

    # Добавляем заголовок Introduction
    P("1. Introduction", "h1")

    # Добавляем первый абзац введения
    P("Loan default prediction is important because lenders need to decide whether an applicant is likely to repay "
      "a loan. In this project, I treat this as a binary classification problem, where the model predicts either "
      "that an applicant will repay the loan or that they may default.", "body")

    # Добавляем описание цели проекта
    P("My aim is to build and compare several classification models using the applicant's financial, demographic "
      "and credit-history information. I am also interested in comparing their accuracy with how easy their "
      "decisions are to understand.", "body")

    # Добавляем описание используемой методологии и метрик
    P("I followed the CRISP-DM methodology throughout the project. This included understanding the business "
      "problem, exploring and preparing the data, training the models and evaluating their results. Since the "
      "dataset contains more non-default cases than default cases, accuracy alone does not give a complete picture "
      "of model performance. For this reason, I also used precision, recall, F1-score and ROC-AUC. Recall is "
      "especially important here because incorrectly classifying someone who later defaults could be more costly "
      "for a lender than incorrectly rejecting an applicant who would have repaid the loan.", "body")

    # Добавляем заголовок раздела Dataset
    P("2. Dataset", "h1")

    # Добавляем описание набора данных
    P("The study uses the publicly available <b>Credit Risk Dataset</b> from Kaggle "
      "(<a href='https://www.kaggle.com/datasets/laotse/credit-risk-dataset' color='#4C72B0'>"
      "kaggle.com/datasets/laotse/credit-risk-dataset</a>). It contains <b>32,581 records</b> and 12 attributes "
      "from demographic, financial to credit-bureau information. The target variable is <b>loan_status</b>, "
      "where 1 is a default (bad credit risk) and 0 a non-default (good credit risk).", "body")


    # Создаём данные для таблицы с описанием признаков
    feat = [
        ["Attribute", "Description", "Type"],
        ["person_age", "Applicant age (years)", "Numeric"],
        ["person_income", "Annual income", "Numeric"],
        ["person_home_ownership", "RENT / OWN / MORTGAGE / OTHER", "Categorical"],
        ["person_emp_length", "Employment length (years)", "Numeric"],
        ["loan_intent", "Purpose of the loan (6 categories)", "Categorical"],
        ["loan_grade", "Loan grade A (best) to G (worst)", "Ordinal"],
        ["loan_amnt", "Loan amount requested", "Numeric"],
        ["loan_int_rate", "Interest rate (%)", "Numeric"],
        ["loan_percent_income", "Loan amount as fraction of income", "Numeric"],
        ["cb_person_default_on_file", "Historical default on record (Y/N)", "Categorical"],
        ["cb_person_cred_hist_length", "Length of credit history (years)", "Numeric"],
        ["loan_status", "TARGET: 1 = default, 0 = non-default", "Binary"],
    ]

    # Создаём таблицу признаков и задаём ширину столбцов
    t = Table(feat, colWidths=[USABLE * 0.30, USABLE * 0.50, USABLE * 0.20])

    # Настраиваем внешний вид таблицы
    t.setStyle(TableStyle([

            # Оформляем заголовок таблицы
            ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),

            # Устанавливаем размер шрифта
            ("FONTSIZE", (0, 0), (-1, -1), 8.5),

            # Используем обычный шрифт для остальных строк
            ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),

            # Чередуем фон строк
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#EEF2F7")]),

            # Выделяем последнюю строку с целевой переменной
            ("BACKGROUND", (0, -1), (-1, -1), colors.HexColor("#FCE9D6")),
            ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),

            # Добавляем границы ячеек
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),

            # Выравниваем текст по центру по вертикали
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),

            # Добавляем внутренние отступы
            ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
            ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ]))

    # Добавляем таблицу в документ
    S.append(t)

    # Добавляем подпись под таблицей
    S.append(Paragraph("Table 1. Dataset attributes.", styles["cap"]))

    # Добавляем заголовок Methodology
    P("3. Methodology", "h1")

    # Добавляем подраздел Exploratory Analysis
    P("3.1 Data Understanding &amp; Exploratory Analysis", "h2")

    # Добавляем описание найденных проблем в данных
    P("Before training the models, I explored the dataset to understand its structure and identify any problems that "
      "needed to be addressed. I found missing values in loan interest rate, affecting approximately 9.6% of the "
      "rows, and in employment length, affecting around 2.7% of the rows. There were also 165 duplicate records. "
      "In addition, some values were not realistic, such as ages up to 144 years and employment lengths up to 123 "
      "years.", "body")

    # Добавляем описание дисбаланса классов
    P("The target variable was imbalanced, with approximately 78% of the applications being non-defaults and 22% "
      "being defaults. Because of this imbalance, I decided to use recall, F1-score and ROC-AUC alongside accuracy "
      "when evaluating the models.", "body")

    # Добавляем первый график с распределением целевой переменной
    fig1 = img("eda_1_target.png", USABLE * 0.85)

    # Выравниваем изображение по центру
    fig1.hAlign = "CENTER"

    # Добавляем изображение в документ
    S.append(fig1)

    # Добавляем подпись к первому графику
    S.append(Paragraph("Figure 1. The distribution of the target variable. Most applications are classified as good "
                       "credit risks, while around 22% are defaults.", styles["cap"]))


    # Добавляем график категориальных признаков
    fig2 = img("eda_3_categorical.png", USABLE * 0.92)

    # Выравниваем изображение по центру
    fig2.hAlign = "CENTER"

    # Добавляем изображение
    S.append(fig2)

    # Добавляем подпись ко второму графику
    S.append(Paragraph("Figure 2. Default rates across the categorical variables. Loan grade appears to be the strongest "
                       "indicator: the default rate increases from approximately 10% for grade A to 98% for grade G. "
                       "Renters also have a considerably higher default rate than homeowners.", styles["cap"]))


    # Добавляем график числовых зависимостей
    S.append(img("eda_4_drivers.png"))

    # Добавляем подпись к третьему графику
    S.append(Paragraph("Figure 3. The default rate increases as the loan-to-income ratio and interest rate increase. "
                       "Among the numeric variables, loan-to-income ratio, with a correlation of 0.38, and interest "
                       "rate, with a correlation of 0.34, have the strongest relationships with the target.", styles["cap"]))

    # Добавляем вывод по результатам EDA
    P("Overall, the exploratory analysis suggested that default risk does not follow a simple straight-line pattern. "
      "Instead, there are noticeable changes across loan grades and loan-to-income bands. This supported my decision "
      "to include tree-based and neural-network models alongside Logistic Regression.", "body")

    # Добавляем подраздел Data Preparation
    P("3.2 Data Preparation", "h2")

    # Добавляем описание этапов подготовки данных
    P("The following steps were applied, with all imputation and scaling parameters fitted on the training set "
      "only to avoid data leakage: (i) removal of 165 duplicate rows and the physically-impossible outliers; "
      "(ii) imputation of missing <i>loan_int_rate</i> using the median rate per loan grade (grade and rate are "
      "tightly linked) and missing <i>person_emp_length</i> using the median; (iii) ordinal encoding of "
      "<i>loan_grade</i> (A&rarr;G as 0&rarr;6), binary encoding of the Y/N default flag, and one-hot encoding of "
      "<i>person_home_ownership</i> and <i>loan_intent</i>; (iv) a stratified 80/20 train/test split; and "
      "(v) standardisation of features for the models that require it. The Logistic Regression and neural network "
      "use standardised inputs, while the Decision Tree uses the raw (scale-invariant) values.", "body")

    # Добавляем подраздел Modelling
    P("3.3 Modelling", "h2")

    # Добавляем общее описание моделей
    P("I trained three different models. I used the same training and test data for "
      "each model so that their results could be compared fairly.", "body")

    # Добавляем описание Logistic Regression
    P("First, I trained a Logistic Regression model. I used this as a baseline because it is a simple model and its "
      "results are relatively easy to understand. It also helped me see how much better the other models performed.", "body")

    # Добавляем описание Decision Tree
    P("The second model was a Decision Tree with a maximum depth of six. I chose this model because it can show its "
      "decisions through a series of rules. This makes it easier to understand why the model classifies an applicant "
      "as more or less likely to default.", "body")

    # Добавляем описание нейронной сети
    P("Finally, I trained an Artificial Neural Network with two hidden layers. The layers contained 32 and 16 units. "
      "I included this model because it can identify more complicated patterns in the data. I also used early "
      "stopping so that the model would not continue training if its performance on new data stopped improving.", "body")

    # Добавляем раздел Results and Evaluation
    P("4. Results and Evaluation", "h1")

    # Добавляем описание результатов Logistic Regression
    P("The results for all three models are shown in Table 2. Overall, the Decision Tree and the neural network "
      "performed much better than Logistic Regression. Logistic Regression achieved an accuracy of about 85%, but "
      "its recall was only 0.50. This means that it identified only about half of the applicants who actually "
      "defaulted.", "body")

    # Добавляем сравнение Decision Tree и neural network
    P("The Decision Tree achieved an accuracy of 91.96%, while the neural network achieved 92.01%. The Decision Tree "
      "had the highest precision, meaning that most of the applicants it classified as potential defaulters really "
      "were defaults. The neural network performed better in terms of recall, F1-score and ROC-AUC, although the "
      "difference between the two models was quite small.", "body")

    # Добавляем таблицу с метриками
    S.append(results_table())

    # Добавляем подпись к таблице
    S.append(Paragraph("Table 2. Performance of the three models on the test set. The best result for each metric is "
                       "highlighted.", styles["cap"]))

    # Добавляем confusion matrices
    S.append(img("model_confusion.png"))

    # Добавляем подпись
    S.append(Paragraph("Figure 4. Confusion matrices for the three models. Logistic Regression missed more defaults "
                       "than the Decision Tree and the neural network.", styles["cap"]))


    # Добавляем ROC-кривые
    fig5 = img("model_roc.png", USABLE * 0.62)

    # Выравниваем график по центру
    fig5.hAlign = "CENTER"

    # Добавляем изображение в документ
    S.append(fig5)

    # Добавляем подпись к ROC-кривым
    S.append(Paragraph("Figure 5. ROC curves for the three models. The neural network achieved the highest ROC-AUC, "
                       "followed by the Decision Tree.", styles["cap"]))

    # Добавляем описание наиболее важных признаков
    P("The Decision Tree also made it possible to look at the rules behind its predictions. The most important "
      "features were the loan-to-income ratio, loan grade and whether the applicant rented their home. This was "
      "similar to what I found during the exploratory analysis. In general, applicants with a high loan-to-income "
      "ratio who rented were more likely to be classified as defaulters. Applicants with a lower loan-to-income "
      "ratio, a better loan grade and sufficient income were more likely to be predicted as non-defaults.", "body")

    # Добавляем раздел Discussion
    P("5. Discussion", "h1")

    # Добавляем обсуждение Logistic Regression
    P("When I compared the three models, the main thing I noticed was that Logistic Regression was not able to "
      "capture all the patterns in the data. It was the simplest model to understand, but it missed quite a lot of "
      "the applicants who actually defaulted. This suggests that the connection between the applicant information "
      "and loan default is not completely linear.", "body")

    # Добавляем обсуждение Decision Tree
    P("The Decision Tree worked much better and was still relatively easy to follow. I could look at its rules and "
      "get an idea of why an applicant was classified as a likely defaulter. This makes the model easier to explain "
      "than the neural network.", "body")

    # Добавляем обсуждение neural network
    P("The neural network gave the strongest results for recall, F1-score and ROC-AUC, but it was much harder to see "
      "how it reached its decisions. Personally, I think the Decision Tree is a good compromise for this project. "
      "Its results were close to those of the neural network, but its predictions were easier to understand. However, "
      "if the main priority was to find as many possible defaulters as possible, the neural network would probably "
      "be the better option.", "body")

    # Добавляем раздел Conclusion and Future Work
    P("6. Conclusion and Future Work", "h1")

    # Добавляем общий вывод проекта
    P("In this project, I used the CRISP-DM process to explore the Kaggle Credit Risk Dataset and compare three "
      "different models for predicting loan default. The results showed that the Decision Tree and the neural "
      "network performed better than Logistic Regression. This suggests that loan default is influenced by several "
      "factors working together, rather than by one simple pattern.", "body")
    
    # Добавляем вывод о наиболее важных признаках
    P("The most important factors appeared to be loan grade, the loan-to-income ratio and home-ownership status. "
      "Based on the results, I think the Decision Tree is the most suitable model for this project because it "
      "performed well while still being possible to understand and explain. The neural network would be useful if "
      "identifying as many potential defaults as possible was the main goal.", "body")

    # Добавляем предложения для будущего улучшения проекта
    P("There are several ways this project could be improved in the future. I could try methods such as class "
      "weighting or SMOTE to deal with the imbalance in the target variable. I could also tune the models more "
      "carefully using cross-validation and compare them with methods such as Random Forest or gradient boosting. "
      "Another useful improvement would be to adjust the decision threshold depending on whether missing a potential "
      "defaulter or incorrectly rejecting a good applicant is considered more costly.", "body")

    # Добавляем заголовок списка источников
    P("Online Resources and References", "h1")

    # Создаём список используемых источников
    refs = [
        "Project source code. GitHub. "
        "<a href='https://github.com/EvgeniiaEliseeva/Machine_learning_project' color='#4C72B0'>"
        "https://github.com/EvgeniiaEliseeva/Machine_learning_project</a>",

        "Credit Risk Dataset. Kaggle. "
        "<a href='https://www.kaggle.com/datasets/laotse/credit-risk-dataset' color='#4C72B0'>"
        "https://www.kaggle.com/datasets/laotse/credit-risk-dataset</a>",

        "F. Pedregosa et al. Scikit-learn: Machine Learning in Python. "
        "<a href='https://scikit-learn.org' color='#4C72B0'>https://scikit-learn.org</a>",

        "The pandas development team. pandas. "
        "<a href='https://pandas.pydata.org' color='#4C72B0'>https://pandas.pydata.org</a>",

        "J. D. Hunter. Matplotlib. "
        "<a href='https://matplotlib.org' color='#4C72B0'>https://matplotlib.org</a>; "
        "M. Waskom. seaborn. "
        "<a href='https://seaborn.pydata.org' color='#4C72B0'>https://seaborn.pydata.org</a>",

        "P. Chapman et al. CRISP-DM 1.0: Step-by-step data mining guide. 2000.",
    ]

    # Добавляем каждый источник в документ с номером
    for i, r in enumerate(refs, 1):
        S.append(Paragraph(f"[{i}]&nbsp;&nbsp;{r}", styles["ref"]))

    # Функция добавляет нижний колонтитул на каждую страницу
    def footer(canvas, doc):

        # Сохраняем текущее состояние оформления страницы
        canvas.saveState()

        # Устанавливаем шрифт и цвет текста
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GREY)

        # Добавляем номер страницы по центру
        canvas.drawCentredString(PAGE_W / 2, 12 * mm, f"{doc.page}")

        # Добавляем название проекта внизу слева
        canvas.drawString(LMARGIN, 12 * mm, "Loan Default Prediction — Project Report")

        # Устанавливаем цвет разделительной линии
        canvas.setStrokeColor(colors.HexColor("#DDDDDD"))

        # Рисуем линию над нижним колонтитулом
        canvas.line(LMARGIN, 16 * mm, PAGE_W - RMARGIN, 16 * mm)

        # Возвращаем исходное состояние страницы
        canvas.restoreState()


    # Создаём итоговый PDF-документ
    doc = SimpleDocTemplate("Loan_Default_Prediction_Report.pdf", pagesize=A4,
                            leftMargin=LMARGIN,rightMargin=RMARGIN,
                            topMargin=18 * mm, bottomMargin=20 * mm,
                            title="Loan Default Prediction - Project Report",
                            author=", ".join(a.split("  (")[0] for a in AUTHORS))

    # Собираем все элементы и создаём PDF-файл. footer добавляется как на первую, так и на остальные страницы.
    doc.build(S, onFirstPage=footer, onLaterPages=footer)

    # Выводим сообщение после успешного создания PDF
    print("PDF built: Loan_Default_Prediction_Report.pdf")


# Запускаем функцию build(), если файл был запущен напрямую
if __name__ == "__main__":
    build()
