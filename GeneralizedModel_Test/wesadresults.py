import itertools
import os
import pickle
from collections import Counter

import numpy as np
import pandas as pd
import scipy.stats
from sklearn.metrics import accuracy_score, f1_score, classification_report

WESAD_SUBJECTS = list(itertools.chain(range(2, 12), range(13, 18)))


def datasets_metrics():
    results = []

    for dataset in ["WESAD_15_fold"]:
        # setups = [f"it_{it:02d}" for it in range(5)]
        setups = [f"it_{it:02d}" for it in range(1)]
        add_baseline(dataset, results)

        for architecture in ['fcnM', 'mlpLstmM', 'resnetM']:
            # for eval_i in range(10):
            for eval_i in range(1):
                results += get_result(architecture, dataset, eval_i, setups)
    return pd.DataFrame(results, columns=["Dataset", "Architecture", "Fold", "Evaluation", "Loss", "Loss (std)", "Accuracy", "Accuracy (std)", "F1", "F1 (std)", "AUC", "AUC (std)", "Duration", "Duration (std)"])


def add_baseline(dataset, results):
    dataset, n, _ = dataset.split("_")
    for fold_i in range(int(n)):
        y_true = []
        for path in paths_with_results_generator("fcnM", dataset, 0, fold_i, int(n), ["it_00"]):
            with open(f"{path}/predictions.txt") as f:
                y_true += [int(x) for x in f.readline().split()]
        add_majority_baseline(dataset, results, y_true, fold_i)
        add_random_baseline(dataset, results, y_true, fold_i)


def add_random_baseline(dataset, results, y_true, fold_i):
    counter = Counter(y_true)
    accuracy = recall = 1 / len(counter)
    precisions = [x[1] / len(y_true) for x in Counter(y_true).items()]
    f1s = [2 * precision * recall / (precision + recall) for precision in precisions]
    f1 = np.mean(f1s)
    results.append([dataset, "Random guess", fold_i, 0, 0, 0, accuracy, np.nan, f1, np.nan])


def add_majority_baseline(dataset, results, y_true, fold_i):
    # y_pred = len(y_true) * [scipy.stats.mode(y_true).mode[0]]
    y_pred = len(y_true) * [scipy.stats.mode(y_true)[0]]
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')
    results.append([dataset, "Majority class", fold_i, 0, 0, 0, accuracy, np.nan, f1, np.nan])


def get_result(architecture, dataset, eval_i, setups):
    dataset, n, _ = dataset.split("_")
    results = []

    for fold_i in range(int(n)):
        loss = []
        accuracy = []
        f1 = []
        auc = []
        duration = []

        for path in paths_with_results_generator(architecture, dataset, eval_i, fold_i, n, setups):
            if not os.path.exists(path + "df_metrics.csv"):
                print(path)
                return [dataset, architecture, eval_i, float("inf"), 0, 0, 0, 0, 0]
            df_metrics = pd.read_csv(path + "df_metrics.csv")
            df_best_model = pd.read_csv(path + "df_best_model.csv")

            loss.append(df_best_model["best_model_val_loss"][0])
            accuracy.append(df_metrics["accuracy"][0])
            f1.append(df_metrics["f1"][0])
            auc.append(1-df_metrics["auc"][0])
            duration.append(df_metrics["duration"][0])

        duration = np.array(duration) / 60
        results.append([dataset, architecture, fold_i, eval_i, np.mean(loss), np.std(loss), np.mean(accuracy),
                        np.std(accuracy), np.mean(f1), np.std(f1), np.mean(auc), np.std(auc), np.mean(duration),
                        np.std(duration)])
    return results


def paths_with_results_generator(architecture, dataset, eval_i, fold_i, folds_n, setups):
    for setup in setups:
        yield f"results/{dataset}_{folds_n}fold_{fold_i:02d}/tune_{eval_i:02d}/{architecture}/{setup}/"


def count_classes_representation():
    counts = {}
    results = []

    for dataset in ["WESAD"]:
        counts[dataset] = []
        for subject in range(100):
            path = f"archives/mts_archive/{dataset}/y_{subject}.pkl"
            if not os.path.exists(path):
                continue
            counts[dataset] += pickle.load(open(path, "rb"))

        counts[dataset] = Counter(counts[dataset])

        line = [dataset]
        # for i in range(1, 5):
        for i in range(1, 4):
            line.append(counts[dataset][i])
        results.append(line)

    # df = pd.DataFrame(results, columns=["Dataset", "LALV", "LAHV", "HALV", "HAHV"])
    df = pd.DataFrame(results, columns=["Dataset", "Baseline", "Stress", "Amuesement"])
    return df


def count_test_classes_representation():
    results = []

    for dataset in ["WESAD"]:
        y_num = []
        result_path = "./results"
        folder_names = os.listdir(result_path)
        folder_names.sort()

        for folder_name in folder_names:
            if folder_name.startswith("WESAD_15fold_"):
                path = os.path.join(result_path, folder_name, "tune_00/fcnM/it_00/predictions.txt")
                if not os.path.exists(path):
                    continue
                with open(path, 'r') as f:
                    lines = f.readlines()
                    y_num.append(len(lines[0].strip().split()))
                    
    results.append(["WESAD", np.mean(y_num), np.std(y_num)])

    df = pd.DataFrame(results, columns=["Dataset", "Num of Test Data (mean)", "Num of Test Data (std)"])
    return df


def rename_architectures(dataset_column):
    dataset_column = dataset_column.str.replace(r'M$', '')
    dataset_column = dataset_column.str.capitalize()
    dataset_column = dataset_column.str.replace(r'Fcn', 'FCN')
    return dataset_column


def prepare_latex_for_paper(latex, caption, label):
    latex = latex.replace(r"nan", " --")
    latex = f"""\\begin{{table*}}
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex}\\end{{table*}}
"""
    return latex


def classification_metrics_for_evaluation(dataset: str, eval_list: list, architecture: str):
    metrics = ["precision", "recall", "f1-score", "support"]
    aggregated_classification_reports = {}

    for fold_i, eval_i in enumerate(eval_list):
        # for setup_path in paths_with_results_generator(architecture, dataset, eval_i, fold_i, 5,
        for setup_path in paths_with_results_generator(architecture, dataset, eval_i, fold_i, 15,
                                                    #    [f"it_{it:02d}" for it in range(5)]):
                                                       [f"it_{it:02d}" for it in range(1)]):
            with open(f"{setup_path}/predictions.txt") as f:
                y_true = [int(x) for x in f.readline().split()]
                if len(aggregated_classification_reports) == 0:
                    for clas in set(y_true):
                        aggregated_classification_reports[clas] = {"f1-score": [], "precision": [], "recall": [],
                                                                   "support": []}

                y_pred = [int(x) for x in f.readline().split()]
                report = classification_report(y_true, y_pred, output_dict=True)
                for clas in aggregated_classification_reports:
                    for metric in metrics:
                        aggregated_classification_reports[clas][metric].append(report[str(clas)][metric])

    result = []
    for clas in aggregated_classification_reports:
        row = [dataset, clas]
        for metric in metrics:
            metric_values = aggregated_classification_reports[clas][metric]
            row.append(np.mean(metric_values))
            row.append(np.std(metric_values))
        result.append(row[:-1])  # Remove support std

    return result


def print_classification_metrics_for_classes(results, evaluation_df):
    metrics = []
    for dataset in results.Dataset.unique():
        results_for_dataset = results[
            (results.Dataset == dataset) & (results.Architecture != "Random guess") & (
                    results.Architecture != "Majority class")]
        best_architecture = results.iloc[results_for_dataset["F1-score"].idxmax()]

        metrics += classification_metrics_for_evaluation(dataset,
                                                         evaluation_df[dataset, best_architecture["Architecture"]],
                                                         best_architecture["Architecture"])

    metrics = pd.DataFrame(metrics, columns=["Dataset", "Class", "Precision", "Precision (std)", "Recall",
                                             "Recall (std)", "F1-score", "F1-score (std)", "Support"])

    # metrics.Class = metrics.Class.apply(lambda x: ["LALV", "LAHV", "HALV", "HAHV"][x])
    metrics.Class = metrics.Class.apply(lambda x: ["Baseline", "Stress", "Amuesement"][x])

    with pd.option_context("display.float_format", "{:,.2f}".format):
        columns = [0, 1, 6, 2, 4, 8]
        column_format = "|l|l" + ((len(columns) - 2) * "|r") + "|"
        caption = "Classification metrics for classes"
        label = "tab:metricsForClasses"

        latex = metrics.iloc[:, columns].to_latex(index=False, column_format=column_format)
        latex = prepare_latex_for_paper(latex, caption, label)
        print(latex)

def print_classification_metrics_for_LOSO(results):
    temp_results = results.sort_values("Fold", ascending=True).groupby(["Dataset", "Architecture"])
    temp_results = temp_results.mean().drop(columns=["Fold", "Evaluation"]).reset_index()
    # metrics = []
    for dataset in temp_results.Dataset.unique():
        results_for_dataset = temp_results[
            (temp_results.Dataset == dataset) & (temp_results.Architecture != "Random guess") & (
                    temp_results.Architecture != "Majority class")]
        best_architecture = temp_results.iloc[results_for_dataset["F1-score"].idxmax()]["Architecture"]
    
    best_results = results[(results.Architecture == best_architecture) | (results.Architecture == "Majority class")]
    best_results = best_results[["Architecture", "Fold", "Accuracy", "F1-score", "ROC AUC"]]

    with pd.option_context("display.float_format", "{:,.2f}".format):
        latex = prepare_latex_for_paper(best_results.to_latex(index=False, column_format="|l|l|r|r|r|"),
                                        f"Best performing model for WESAD in detail", f"tab:datasetsClassesLOSO")
                
        print(latex)

def metrics_for_best_evaluations():
    results = datasets_metrics()
    best = []
    for dataset in results.Dataset.unique():
        for architecture in results.Architecture.unique():
            for fold in results.Fold.unique():
                results_for_dataset_arch = results[(results.Dataset == dataset) & (results.Architecture == architecture)
                                                   & (results.Fold == fold)]
                min_loss_id = results_for_dataset_arch["Loss"].idxmin()
                best_loss_result = results.iloc[min_loss_id]
                best.append(list(best_loss_result))

    return pd.DataFrame(best, columns=["Dataset", "Architecture", "Fold", "Evaluation", "Validation loss",
                                       "Validation loss (std)", "Accuracy", "Accuracy (std)", "F1-score",
                                       "F1-score (std)", "ROC AUC", "ROC AUC (std)", "Duration (min)",
                                       "Duration (std)"])


def prepare_readable_values(results):
    results.Architecture = rename_architectures(results.Architecture)
    # results["Duration (min)"] = results["Duration (min)"].map('{:,.1f}'.format)
    # results["Duration (std)"] = results["Duration (std)"].map("{:,.1f}".format)
    return results


def create_file_for_cd_diagram(results, dataset_name):
    results[(results.Architecture != "Random guess") & (
            results.Architecture != "Majority class")][["Architecture", "Dataset", "F1-score"]].to_csv(
        f"csvresults/{dataset_name}/resultsForCriticalDiffrencesDiagram.csv", index=False)
    
def create_file_for_LOSO(results, dataset_name):
    results[["Dataset", "Architecture", "Fold", "Evaluation", "Validation loss",
"Validation loss (std)", "Accuracy", "Accuracy (std)", "F1-score",
"F1-score (std)", "ROC AUC", "ROC AUC (std)", "Duration (min)", "Duration (std)"]].to_csv(
       f"csvresults/{dataset_name}/resultsForLOSO.csv", index=False)


def print_metrics_for_datasets():
    for dataset in results.Dataset.unique():
        with pd.option_context("display.float_format", "{:,.2f}".format):
            latex = results[results.Dataset == dataset]. \
                sort_values("F1-score", ascending=False)[
                ["Architecture", "Accuracy", "Accuracy (std)", "F1-score", "F1-score (std)", "ROC AUC",
                 "ROC AUC (std)"]].to_latex(index=False, column_format="|l|r|r|r|r|r|r|r|r|")
            latex = prepare_latex_for_paper(latex, f"Results for {dataset} ordered by averaged F1-score",
                                            f"tab:{dataset}Results")
            print(latex)


def print_classes_representation():
    df = count_classes_representation()
    latex = prepare_latex_for_paper(df.to_latex(index=False, column_format="|l|r|r|r|r|"),
                                    f"Classes balance in datasets", f"tab:datasetsClassesBalance")
    # # This prints out the exact number of data in each class
    # print(latex)

    df = df.set_index("Dataset")
    res = df.div(df.sum(axis=1), axis=0) * 100
    with pd.option_context("display.float_format", "{:,.0f}%".format):
        latex = prepare_latex_for_paper(res.reset_index().to_latex(index=False, column_format="|l|r|r|r|r|"),
                                        f"Classes balance in datasets in percentage", f"tab:datasetsClassesBalancePerc")
        print(latex)

def print_test_classes_representation():
    df = count_test_classes_representation()
    with pd.option_context("display.float_format", "{:,.0f}%".format):
        latex = prepare_latex_for_paper(df.to_latex(index=False, column_format="|l|r|r|r|"),
                                        f"Number of Test Data", f"tab:testdatanum")
        print(latex)


if __name__ == '__main__':
    results = metrics_for_best_evaluations()
    create_file_for_LOSO(results, "WESAD")

    # # This prints out detailed LOSO classification metrics for the best performing (highest F1 score) model
    # print_classification_metrics_for_LOSO(results)

    results = results.sort_values("Fold", ascending=True).groupby(["Dataset", "Architecture"])
    # evaluation_df = results["Evaluation"].apply(list)

    results = results.agg({
        "Accuracy": ['mean', 'std'],
        "F1-score": ['mean', 'std'],
        "ROC AUC": ['mean', 'std'],
    }).reset_index()

    results = pd.DataFrame({
        'Dataset': results['Dataset'],
        'Architecture': results['Architecture'],
        'Accuracy': results['Accuracy', 'mean'],
        'Accuracy (std)': results['Accuracy', 'std'],
        'F1-score': results['F1-score', 'mean'],
        'F1-score (std)': results['F1-score', 'std'],
        'ROC AUC': results['ROC AUC', 'mean'],
        'ROC AUC (std)': results['ROC AUC', 'std']
    }
    )

    # # This prints out classification metrics for each class using the best performing (highest F1 score) model
    # print_classification_metrics_for_classes(results, evaluation_df)

    results = prepare_readable_values(results)

    create_file_for_cd_diagram(results, "WESAD")

    print_metrics_for_datasets()

    # print_classes_representation()
    
    print_test_classes_representation()