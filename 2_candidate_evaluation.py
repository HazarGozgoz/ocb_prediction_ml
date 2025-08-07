import argparse
import pandas as pd
import numpy as np
import os
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, f1_score, roc_curve, precision_recall_curve, \
    auc, precision_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import shap

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- AYARLAR ---
plt.rcParams["savefig.format"] = "tiff";
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.family"] = "Times New Roman";
plt.rcParams["font.size"] = 12
RANDOM_STATE = 42;
np.random.seed(RANDOM_STATE);
random.seed(RANDOM_STATE)
warnings.filterwarnings("ignore")
OUTPUT_DIR = "results/Final_Algorithm_Comparison_v2"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "plots"), exist_ok=True)


# --- YARDIMCI FONKSİYONLAR ---

def load_and_prepare_data(file_path):
    data = pd.read_excel(file_path)
    data = data.loc[:, data.notna().any()]
    data = data.dropna(subset=['Oligoklonal bant'])
    data['Oligoklonal bant'] = data['Oligoklonal bant'].astype(int)
    if 'BOS Total Protein' in data.columns and 'BOS Albumin' in data.columns: data["BOS_Protein_to_BOS_Albumin_Ratio"] = \
    data['BOS Total Protein'] / (data['BOS Albumin'] + 1e-6)
    if 'BOS IgG' in data.columns: data["Log_IgG_BOS"] = np.log1p(data['BOS IgG'])
    if 'Serum IgG' in data.columns: data["Log_Serum_IgG"] = np.log1p(data['Serum IgG'])
    if 'BOS Sodyum' in data.columns and 'BOS Potasyum' in data.columns: data["Sodium_Potassium_Diff"] = data[
                                                                                                            'BOS Sodyum'] - \
                                                                                                        data[
                                                                                                            'BOS Potasyum']
    if 'BOS Glukoz' in data.columns and 'BOS Total Protein' in data.columns: data["Glucose_to_Protein_Ratio"] = data[
                                                                                                                    'BOS Glukoz'] / (
                                                                                                                            data[
                                                                                                                                'BOS Total Protein'] + 1e-6)
    if 'BOS Albumin' in data.columns and 'Serum Albumin' in data.columns: data["CSF_Serum_Albumin_Ratio"] = data[
                                                                                                                "BOS Albumin"] / (
                                                                                                                        data[
                                                                                                                            "Serum Albumin"] + 1e-6)
    if "YAŞ" in data.columns: data.rename(columns={"YAŞ": "YAS"}, inplace=True)
    print("✅ Data loading and feature engineering completed.")
    return data


# ⭐️ GÜNCELLENDİ: PR-AUC eklendi
def get_bootstrap_metric_distributions(y_true, y_pred_prob, n_iterations=1000):
    y_true, y_pred_prob = np.array(y_true), np.array(y_pred_prob)
    metric_scores = {
        "ROC-AUC": [], "PR-AUC": [], "Accuracy": [], "Sensitivity": [], "Specificity": [],
        "Precision (Positive)": [], "Precision (Negative)": [], "F1 (Positive)": [], "F1 (Negative)": []
    }
    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_true)), replace=True, random_state=None)
        if len(np.unique(y_true[indices])) < 2: continue
        y_true_boot, y_pred_prob_boot = y_true[indices], y_pred_prob[indices]

        fpr, tpr, thresholds = roc_curve(y_true_boot, y_pred_prob_boot)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
        y_pred_boot = (y_pred_prob_boot >= optimal_threshold).astype(int)

        # PR-AUC hesaplaması
        precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_prob_boot)
        metric_scores["PR-AUC"].append(auc(recall, precision))

        metric_scores["ROC-AUC"].append(roc_auc_score(y_true_boot, y_pred_prob_boot))
        precision, recall, _ = precision_recall_curve(y_true_boot, y_pred_prob_boot)
        metric_scores["Accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
        metric_scores["Sensitivity"].append(recall_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metric_scores["Specificity"].append(recall_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
        metric_scores["F1 (Positive)"].append(f1_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metric_scores["F1 (Negative)"].append(f1_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
        metric_scores["Precision (Positive)"].append(
            precision_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metric_scores["Precision (Negative)"].append(
            precision_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))

    for key in metric_scores: metric_scores[key] = np.array(metric_scores[key])
    return metric_scores


def format_ci_string(scores):
    mean, lower, upper = np.mean(scores), np.percentile(scores, 2.5), np.percentile(scores, 97.5)
    return f"{mean:.3f} ({lower:.3f}–{upper:.3f})"


def perform_statistical_tests(bootstrap_scores_dict):
    models = list(bootstrap_scores_dict.keys())
    p_values_data = []
    print("\n--- Pairwise Statistical Significance (p-values for ROC-AUC) ---")
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            model1, model2 = models[i], models[j]
            scores1, scores2 = bootstrap_scores_dict[model1]["ROC-AUC"], bootstrap_scores_dict[model2]["ROC-AUC"]
            diff_scores = scores1 - scores2
            p_value = 2 * np.mean(diff_scores < 0) if np.mean(diff_scores) > 0 else 2 * np.mean(diff_scores > 0)
            p_value_str = "< 0.001" if p_value == 0 else f"{p_value:.4f}"
            print(f"{model1} vs {model2}: p = {p_value_str}")
            p_values_data.append({"Comparison": f"{model1} vs {model2}", "p-value": p_value_str})
    return pd.DataFrame(p_values_data)


def plot_cv_boxplots(cv_results_df):
    plt.figure(figsize=(10, 6));
    sns.boxplot(data=cv_results_df, palette="viridis");
    sns.stripplot(data=cv_results_df, color=".25")
    plt.ylabel("ROC-AUC Score");
    plt.xlabel("Model");
    plt.title("Distribution of Model Performance Across 5 Folds")
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "cv_performance_boxplot.tiff"));
    plt.show()


def plot_all_roc_curves(roc_data):
    plt.figure(figsize=(8, 7))
    for name, data in roc_data.items(): plt.plot(data['fpr'], data['tpr'], lw=2,
                                                 label=f"{name} (AUC = {data['auc']:.3f})")
    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance (AUC = 0.50)');
    plt.xlabel("False Positive Rate");
    plt.ylabel("True Positive Rate (Sensitivity)")
    plt.title("Comparison of ROC Curves for All Models");
    plt.legend(loc="lower right");
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout();
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "all_models_roc_curves.tiff"));
    plt.show()


# ⭐️ GÜNCELLENDİ: `display_names_map` argümanı eklendi
def plot_shap_comparisons(models, X, y, feature_names, display_names_map):
    fig, axes = plt.subplots(1, len(models), figsize=(20, 7))
    if len(models) == 1: axes = [axes]
    print("\n--- Generating Comparative SHAP Plots ---")
    for ax, (name, model) in zip(axes, models.items()):
        model.fit(X, y);
        explainer = shap.TreeExplainer(model);
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list): shap_values = shap_values[1]
        mean_abs_shap = np.abs(shap_values).mean(0)
        shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': mean_abs_shap})
        shap_df['Feature'] = shap_df['Feature'].map(display_names_map)

        shap_df = shap_df.sort_values(by='SHAP Value', ascending=False).head(10)
        sns.barplot(x='SHAP Value', y='Feature', data=shap_df, ax=ax, palette='viridis')
        ax.set_title(f"{name} Feature Importance");
        ax.set_xlabel("Mean Absolute SHAP Value")

    axes[0].set_ylabel("Features");
    for ax in axes[1:]: ax.set_ylabel("")
    fig.suptitle("Side-by-Side Comparison of Global Feature Importances (SHAP)", fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.95]);
    plt.savefig(os.path.join(OUTPUT_DIR, "plots", "comparative_shap_summary.tiff"));
    plt.show()


# --- ANA İŞ AKIŞI ---

def advanced_compare_algorithms(file_path):
    data = load_and_prepare_data(file_path)
    selected_features = ['Log_IgG_BOS', 'YAS', "BOS_Protein_to_BOS_Albumin_Ratio", 'Log_Serum_IgG',
                         'Sodium_Potassium_Diff', 'CRP', 'Glucose_to_Protein_Ratio', 'CSF_Serum_Albumin_Ratio']

    feature_display_names = {
        'Log_IgG_BOS': 'log(CSF IgG)',
        'YAS': 'Age',
        'BOS_Protein_to_BOS_Albumin_Ratio': 'CSF Protein / Albumin Ratio',
        'Log_Serum_IgG': 'log(Serum IgG)',
        'Sodium_Potassium_Diff': 'CSF Sodium-Potassium Diff.',
        'CRP': 'C-Reactive Protein (CRP)',
        'Glucose_to_Protein_Ratio': 'CSF Glucose / Protein Ratio',
        'CSF_Serum_Albumin_Ratio': 'CSF / Serum Albumin Ratio'
    }

    X = data[selected_features];
    y = data['Oligoklonal bant']
    imputer = SimpleImputer(strategy='median')
    X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=selected_features)

    models = {
        "XGBoost": XGBClassifier(n_estimators=263, max_depth=3, learning_rate=0.06, subsample=0.8, colsample_bytree=0.8,
                                 random_state=RANDOM_STATE, use_label_encoder=False, eval_metric='logloss'),
        "LightGBM": LGBMClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, num_leaves=30, subsample=0.7,
                                   colsample_bytree=0.7, random_state=RANDOM_STATE, deterministic=True,
                                   force_row_wise=True),
        "CatBoost": CatBoostClassifier(iterations=400, depth=4, learning_rate=0.08, l2_leaf_reg=3.0,
                                       random_seed=RANDOM_STATE, verbose=0)
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    all_results, cv_results_df, bootstrap_scores_dict, roc_data = [], pd.DataFrame(), {}, {}

    for name, model in models.items():
        print(f"--- Evaluating {name} using 5-Fold Cross-Validation ---")
        y_true_cv, y_pred_prob_cv, cv_fold_scores = [], [], []

        for train_idx, test_idx in cv.split(X_imputed, y):
            X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            y_pred_prob_cv.extend(model.predict_proba(X_test)[:, 1])
            y_true_cv.extend(y_test)
            cv_fold_scores.append(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

        cv_results_df[name] = cv_fold_scores
        print(f"Calculating bootstrap metrics for {name}...")
        bootstrap_distributions = get_bootstrap_metric_distributions(y_true_cv, y_pred_prob_cv)
        bootstrap_scores_dict[name] = bootstrap_distributions
        result_row = {'Model': name}
        for metric_name, scores in bootstrap_distributions.items(): result_row[metric_name] = format_ci_string(scores)
        all_results.append(result_row)

        fpr, tpr, _ = roc_curve(y_true_cv, y_pred_prob_cv)
        roc_data[name] = {'fpr': fpr, 'tpr': tpr, 'auc': np.mean(bootstrap_distributions["ROC-AUC"])}

    results_df = pd.DataFrame(all_results)
    ordered_cols = ['Model', 'ROC-AUC', 'PR-AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision (Positive)',
                    'Precision (Negative)', 'F1 (Positive)',
                    'F1 (Negative)']
    results_df = results_df[ordered_cols]

    print("\n--- Advanced Algorithm Comparison (5-Fold CV with 95% CI) ---")
    print(results_df.to_string(index=False))

    excel_path = os.path.join(OUTPUT_DIR, "advanced_algorithm_comparison.xlsx")
    results_df.to_excel(excel_path, index=False)
    print(f"\n✅ Main comparison report saved to: {excel_path}")

    p_values_df = perform_statistical_tests(bootstrap_scores_dict)
    p_values_df.to_excel(os.path.join(OUTPUT_DIR, "statistical_tests.xlsx"), index=False)
    print(f"✅ Statistical test results saved.")

    plot_cv_boxplots(cv_results_df)
    plot_all_roc_curves(roc_data)
    # ⭐️ GÜNCELLENDİ: İsim haritası fonksiyona gönderiliyor
    plot_shap_comparisons(models, X_imputed, y, selected_features, feature_display_names)


if __name__ == '__main__':
    # Komut satırından argüman almak için bir parser oluştur
    parser = argparse.ArgumentParser(
        description="Compare ML algorithms for OCB prediction."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Path to the input Excel data file (e.g., book4.xlsx)"
    )

    args = parser.parse_args()

    # Ana fonksiyonu, komut satırından gelen dosya yolu ile çağır
    screen_algorithms_and_interpret(file_path=args.input_file)