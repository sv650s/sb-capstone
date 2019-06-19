import matplotlib.pyplot as plt
import seaborn as sns
import re
import pandas as pd


# common utilities for various notebooks


# function to print mutltiple histograms
def plot_score_histograms(df: pd.DataFrame):

    if "label" not in df.columns:
        df["label"] = df.apply(lambda x: f'{x["model_name"]}-{x["description"]}', axis=1)

    models = df["model_name"].unique()
    f1_cols, precision_cols, recall_cols = get_score_columns(df)
    for model in models:
        model_report = df[df["label"].str.startswith(f'{model}-')]

        pos = list(range(len(model_report)))
        width = 0.15
        f, a = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(20, len(model_report) * 1))
        column_dict = {"F1": f1_cols, "Precision": precision_cols, "Recall": recall_cols}

        # sort the report in reverse order so we see the models top down
        report_reverse = model_report.sort_values("label", ascending=False)

        index = 0
        for title, columns in column_dict.items():
            columns_copy = columns.copy()
            columns_copy.remove("label")
            # sort in reverse order so it goes top-down and 5 is at the bottom
            columns_copy.sort(reverse=True)

            offset = 0
            for col in columns_copy:
                #         print(f'Plotting {col}')
                a[index].barh([p + offset for p in pos],
                              report_reverse[col],
                              width,
                              #                 align="edge",
                              #                 alpha=0.5,
                              tick_label=report_reverse["label"].tolist(),
                              orientation="horizontal")
                offset += width
                a[index].set_title(title)
                a[index].set_xlim(0, 1.0)
            index += 1


def plot_macro_data(df: pd.DataFrame, cv=False):
    if "label" not in df.columns:
        df["label"] = df.apply(lambda x: f'{x["model_name"]}-{x["description"]}', axis=1)

    df = df.sort_values(["label"])

    f, a = plt.subplots(1, 1, figsize=(20, 5))

    g = sns.lineplot(data=df, x="label", y="macro avg_f1-score", label="avg_f1-score", ax=a, color="r")
    g = sns.lineplot(data=df, x="label", y="macro avg_precision", label="avg_precision", ax=a, color="b")
    g = sns.lineplot(data=df, x="label", y="macro avg_recall", label="avg_recall", ax=a, color="g")
    g.set_xticklabels(labels=df["label"], rotation=90)
    g.set_ylabel("percentage")
    g.set_title("Macro Average Scores And Total Time")
    g.legend(loc="upper left")

    ax2 = a.twinx()

    x = df.label.tolist()
    predict_times = df.total_time_min.tolist()
    fit_times = (df.file_load_time_min + df.train_time_min).tolist()
    if cv:
        cv_times = (df.cv_time_min + df.file_load_time_min + df.train_time_min)
    load_times = df.file_load_time_min.tolist()

    g = sns.barplot(data=df, x="label", y="predict_time_min", label="Predict Time", ax=ax2, color="c", alpha=0.5)
    if cv:
        g = sns.barplot(x=x, y=cv_times, label="CV Time", ax=ax2, color="tab:green", alpha=0.5)
    g = sns.barplot(x=x, y=fit_times, label="Fit Time", ax=ax2, color="tab:orange", alpha=0.5)
    g = sns.barplot(x=x, y=load_times, label="File Load Time", ax=ax2, color="tab:blue", alpha=0.5)
    ax2.tick_params(axis='y', labelcolor="c")
    _ = ax2.legend(loc="upper right")


# function that we will use later
def get_score_columns(df: pd.DataFrame) -> (list, list, list):
    """
    Gets the different score columns from a results DF
    :param df:
    :return:
    """
    CLASS_F1_COLS = [col for col in df.columns if len(re.findall(r'^(\d.+score)', col)) > 0]
    CLASS_F1_COLS.append("label")
    #     print(CLASS_F1_COLS)

    CLASS_PRECISION_COLS = [col for col in df.columns if len(re.findall(r'^(\d+_precision)', col)) > 0]
    CLASS_PRECISION_COLS.append("label")
    #     print(CLASS_PRECISION_COLS)

    CLASS_RECALL_COLS = [col for col in df.columns if len(re.findall(r'^(\d+_recall)', col)) > 0]
    CLASS_RECALL_COLS.append("label")
    #     print(CLASS_RECALL_COLS)

    return CLASS_F1_COLS, CLASS_PRECISION_COLS, CLASS_RECALL_COLS
