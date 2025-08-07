# Gerekli kütüphanelerin import edilmesi
import argparse
import matplotlib

matplotlib.use('TkAgg')  # GUI backend'i olarak TkAgg kullanımı
import pandas as pd
import numpy as np
import shap
import os
import joblib
import warnings
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix, auc
)
from sklearn.utils import resample
from sklearn.calibration import calibration_curve
from sklearn.impute import SimpleImputer

# Model kütüphaneleri
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# --- GÖRSELLEŞTİRME ve TEKRARLANABİLİRLİK AYARLARI ---
plt.rcParams["savefig.format"] = "tiff"
plt.rcParams["figure.dpi"] = 600
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 12

# Tekrarlanabilirlik için random state'lerin sabitlenmesi
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)
os.environ['PYTHONHASHSEED'] = str(RANDOM_STATE)

# Uyarıların gizlenmesi
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Sonuçların kaydedileceği dizinlerin oluşturulması
OUTPUT_DIR = "results/SYNAPSI_yayın4_son"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "shap_plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "model_plots"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "confusion_matrices"), exist_ok=True)


# --- VERİ YÜKLEME VE İŞLEME FONKSİYONLARI ---
def load_data(file_path):
    """Veri setini Excel dosyasından yükler."""
    data = pd.read_excel(file_path)
    print("Data Loaded Successfully")
    print(f"Dataset Shape: {data.shape}")
    return data


def feature_engineering(data):
    """Mevcut özelliklerden yeni özellikler türetir."""
    data = data.copy()
    if 'BOS Total Protein' in data.columns and 'BOS Albumin' in data.columns:
        data["BOS_Protein_to_BOS_Albumin_Ratio"] = data['BOS Total Protein'] / (data['BOS Albumin'] + 1e-6)
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
    print("Feature Engineering Completed")
    return data


# --- METODOLOJİ FONKSİYONLARI ---
def find_optimal_thresholds(y_true, y_pred_prob):
    """F1-Score ve Youden's Index'e göre en iyi eşik değerlerini bulur."""
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_pred_prob)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores[:-1])
    f1_threshold = thresholds_pr[best_f1_idx]
    fpr, tpr, thresholds_roc = roc_curve(y_true, y_pred_prob)
    youden_index = np.argmax(tpr - fpr)
    youden_threshold = thresholds_roc[youden_index]
    return f1_threshold, youden_threshold


def bootstrap_metrics(y_true, y_prob, threshold, n_iterations=1000):
    """Bootstrap yöntemiyle metriklerin %95 güven aralıklarını hesaplar."""
    y_true, y_prob = np.array(y_true), np.array(y_prob)
    np.random.seed(RANDOM_STATE)
    metrics = {"ROC-AUC": [], "PR-AUC": [], "Accuracy": [], "F1-Score (Positive)": [], "Precision (Positive)": [],
               "Recall (Positive)": [],
               "F1-Score (Negative)": [], "Precision (Negative)": [], "Recall (Negative)": []}
    for _ in range(n_iterations):
        indices = resample(np.arange(len(y_true)), replace=True)
        if len(np.unique(y_true[indices])) < 2: continue
        y_true_boot, y_prob_boot = y_true[indices], y_prob[indices]
        y_pred_boot = (y_prob_boot >= threshold).astype(int)
        metrics["ROC-AUC"].append(roc_auc_score(y_true_boot, y_prob_boot))
        precision_b, recall_b, _ = precision_recall_curve(y_true_boot, y_prob_boot)
        metrics["PR-AUC"].append(auc(recall_b, precision_b))
        metrics["Accuracy"].append(accuracy_score(y_true_boot, y_pred_boot))
        metrics["F1-Score (Positive)"].append(f1_score(y_true_boot, y_pred_boot, pos_label=1))
        metrics["Precision (Positive)"].append(precision_score(y_true_boot, y_pred_boot, pos_label=1, zero_division=0))
        metrics["Recall (Positive)"].append(recall_score(y_true_boot, y_pred_boot, pos_label=1))
        metrics["F1-Score (Negative)"].append(f1_score(y_true_boot, y_pred_boot, pos_label=0))
        metrics["Precision (Negative)"].append(precision_score(y_true_boot, y_pred_boot, pos_label=0, zero_division=0))
        metrics["Recall (Negative)"].append(recall_score(y_true_boot, y_pred_boot, pos_label=0))
    return {key: {"mean": np.mean(values), "95% CI": (np.percentile(values, 2.5), np.percentile(values, 97.5))} for
            key, values in metrics.items()}


def ci_to_string(ci_dict):
    """Bootstrap metrik sonuçlarını string formatına dönüştürür."""
    return f"{ci_dict['mean']:.3f} ({ci_dict['95% CI'][0]:.3f}–{ci_dict['95% CI'][1]:.3f})"


def plot_single_shap_summary(model, X_data, model_name, feature_names_map):
    """
    Verilen model için tek bir SHAP özet grafiği oluşturur ve kaydeder.
    """
    output_path = os.path.join(OUTPUT_DIR, "shap_plots")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_data)

    X_data_renamed = X_data.rename(columns=feature_names_map)

    plt.figure()
    shap.summary_plot(shap_values, X_data_renamed, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{model_name}_shap_bar.tiff"), dpi=600)
    plt.close()

    plt.figure()
    shap.summary_plot(shap_values, X_data_renamed, show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"{model_name}_shap_beeswarm.tiff"), dpi=600)
    plt.close()


def plot_combined_curves(y_true, y_prob, model_name):
    """ROC ve PR eğrilerini tek bir figürde birleştirip kaydeder."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc_val = roc_auc_score(y_true, y_prob)
    ax1.plot(fpr, tpr, lw=2, label=f'AUC = {roc_auc_val:.3f}')
    ax1.plot([0, 1], [0, 1], 'k--', lw=2, label='Chance')
    ax1.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel='False Positive Rate',
            ylabel='True Positive Rate (Sensitivity)')
    ax1.set_title("A) Receiver Operating Characteristic Curve", loc='left', fontsize=12)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle='--', alpha=0.6)

    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(recall, precision)
    ax2.plot(recall, precision, lw=2, label=f'AUC = {pr_auc:.3f}')
    ax2.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], xlabel='Recall (Sensitivity)',
            ylabel='Precision (Positive Predictive Value)')
    ax2.set_title("B) Precision-Recall (PR) Curve", loc='left', fontsize=12)
    ax2.legend(loc="lower left")
    ax2.grid(True, linestyle='--', alpha=0.6)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "model_plots", f"roc_pr_curves_{model_name}.tiff"), dpi=600)
    plt.close()


def plot_combined_confusion_matrices(y_true, y_prob, thresholds_dict, filename):
    """
    İki farklı eşik değerine göre konfüzyon matrislerini tek bir figürde birleştirir.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    fig.subplots_adjust(wspace=0.3)

    labels = ['A', 'B']
    for i, ax in enumerate(axes):
        label_key = labels[i]
        threshold = list(thresholds_dict.values())[i]
        y_pred = (np.array(y_prob) >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['OCB-', 'OCB+'], yticklabels=['OCB-', 'OCB+'], ax=ax)

        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        ax.text(0, 1.05, f"{label_key}", transform=ax.transAxes, ha='left', va='bottom', fontsize=12, fontweight='bold')


    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrices", filename), dpi=600)
    plt.close()


def plot_calibration_curve(y_true, y_prob, model_name):
    """Model için sadece kalibrasyon eğrisini çizer ve kaydeder."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=10, strategy='uniform')
    plt.figure(figsize=(6, 5.5))
    plt.plot(prob_pred, prob_true, marker='o', lw=2, label='SYNAPSI Model')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='gray', label='Perfect Calibration')
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives (Observed Frequency)")
    plt.title("Calibration of Predicted Probabilities for OCB Status", loc='left')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    filename = f"calibration_curve_{model_name}.tiff"
    plt.savefig(os.path.join(OUTPUT_DIR, "model_plots", filename), dpi=600)
    plt.close()


# --- ANA İŞ AKIŞI ---
def main(file_path):
    """Tüm model geliştirme, değerlendirme ve raporlama sürecini yönetir."""

    # 1. VERİ YÜKLEME VE ÖN İŞLEME
    data = load_data(file_path)
    data = data.loc[:, data.notna().any()]
    data = data.dropna(subset=['Oligoklonal bant'])
    data['Oligoklonal bant'] = data['Oligoklonal bant'].astype(int)
    data = feature_engineering(data)

    selected_features = [
        'Log_IgG_BOS', 'YAS', "BOS_Protein_to_BOS_Albumin_Ratio", 'Log_Serum_IgG',
        'Sodium_Potassium_Diff', 'CRP', 'Glucose_to_Protein_Ratio', 'CSF_Serum_Albumin_Ratio'
    ]
    target = data['Oligoklonal bant']
    features = data[selected_features]

    feature_display_names = {
        'Log_IgG_BOS': 'Log(CSF IgG)',
        'YAS': 'Age',
        'BOS_Protein_to_BOS_Albumin_Ratio': 'CSF Protein/Albumin Ratio',
        'Log_Serum_IgG': 'Log(Serum IgG)',
        'Sodium_Potassium_Diff': 'CSF Sodium-Potassium Diff.',
        'CRP': 'C-Reactive Protein (CRP)',
        'Glucose_to_Protein_Ratio': 'CSF Glucose/Protein Ratio',
        'CSF_Serum_Albumin_Ratio': 'CSF/Serum Albumin Ratio'
    }

    # 2. OPTİMİZASYONSUZ YAPI İÇİN SABİT PARAMETRELER
    best_params_catboost = {'iterations': 400, 'depth': 4, 'learning_rate': 0.08, 'l2_leaf_reg': 3.0}
    best_params_xgboost = {'n_estimators': 263, 'max_depth': 3, 'learning_rate': 0.06, 'subsample': 0.8,
                           'colsample_bytree': 0.8, 'use_label_encoder': False, 'eval_metric': 'logloss'}
    best_params_lightgbm = {'n_estimators': 150, 'max_depth': 4, 'learning_rate': 0.05, 'num_leaves': 30,
                            'subsample': 0.7, 'colsample_bytree': 0.7, 'force_row_wise': True}

    score_catboost = 0.957
    score_xgboost = 0.949
    score_lightgbm = 0.952

    model_scores = [score_catboost, score_xgboost, score_lightgbm]
    total_score = sum(model_scores)
    dynamic_weights = [score / total_score for score in model_scores]
    print(f"\n✅ Sabit parametrelerle dinamik ağırlıklar hesaplandı.")
    print(f"Dynamically calculated weights: "
          f"CatBoost={dynamic_weights[0]:.3f}, "
          f"XGBoost={dynamic_weights[1]:.3f}, "
          f"LightGBM={dynamic_weights[2]:.3f}")

    # 3. STABİL EŞİK DEĞERLERİNİN BULUNMASI (CatBoost ile)
    print(f"\n--- 📈 Starting 5-fold CV to find stable optimal thresholds using CatBoost ---")

    f1_thresholds, youden_thresholds = [], []
    cv_thresholds = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for train_idx, test_idx in cv_thresholds.split(features, target):
        X_train_fold, X_test_fold = features.iloc[train_idx], features.iloc[test_idx]
        y_train_fold, y_test_fold = target.iloc[train_idx], target.iloc[test_idx]

        imputer_fold = SimpleImputer(strategy='median')
        X_train_fold_imputed = pd.DataFrame(imputer_fold.fit_transform(X_train_fold), columns=features.columns)
        X_test_fold_imputed = pd.DataFrame(imputer_fold.transform(X_test_fold), columns=features.columns)

        temp_model = CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0)
        temp_model.fit(X_train_fold_imputed, y_train_fold)
        y_pred_prob_fold = temp_model.predict_proba(X_test_fold_imputed)[:, 1]

        f1_thresh, youden_thresh = find_optimal_thresholds(y_test_fold, y_pred_prob_fold)
        f1_thresholds.append(f1_thresh)
        youden_thresholds.append(youden_thresh)

    final_f1_threshold = np.median(f1_thresholds)
    final_youden_threshold = np.median(youden_thresholds)
    print("\n✅ Stable optimal thresholds determined:")
    print(f"Final F1 Threshold (Median): {final_f1_threshold:.4f}")
    print(f"Final Youden Threshold (Median): {final_youden_threshold:.4f}")

    # 4. FİNAL ENSEMBLE MODELİN 5-FOLD CROSS-VALIDATION İLE DEĞERLENDİRİLMESİ
    print("\n--- 🧪 Evaluating final model on the entire dataset via 5-Fold Cross-Validation ---")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    y_true_cv, y_pred_prob_cv = [], []

    for train_idx, test_idx in cv.split(features, target):
        X_train_fold, X_test_fold = features.iloc[train_idx], features.iloc[test_idx]
        y_train_fold, y_test_fold = target.iloc[train_idx], target.iloc[test_idx]

        imputer_fold = SimpleImputer(strategy='median')
        X_train_fold_imputed = pd.DataFrame(imputer_fold.fit_transform(X_train_fold), columns=features.columns)
        X_test_fold_imputed = pd.DataFrame(imputer_fold.transform(X_test_fold), columns=features.columns)

        ensemble_model_fold = VotingClassifier(
            estimators=[('CatBoost', CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0)),
                        ('XGBoost', XGBClassifier(**best_params_xgboost, random_state=RANDOM_STATE)),
                        ('LightGBM', LGBMClassifier(**best_params_lightgbm, random_state=RANDOM_STATE))],
            voting='soft',
            weights=dynamic_weights
        )
        ensemble_model_fold.fit(X_train_fold_imputed, y_train_fold)

        y_pred_prob_cv.extend(ensemble_model_fold.predict_proba(X_test_fold_imputed)[:, 1])
        y_true_cv.extend(y_test_fold)

    print("✅ Ensemble model evaluation completed via 5-Fold Cross-Validation.")

    metrics_f1_ci = bootstrap_metrics(y_true_cv, y_pred_prob_cv, final_f1_threshold)
    metrics_youden_ci = bootstrap_metrics(y_true_cv, y_pred_prob_cv, final_youden_threshold)
    metrics_05_ci = bootstrap_metrics(y_true_cv, y_pred_prob_cv, 0.5)

    report_data = {
        "Threshold Type": ["Stable F1", "Stable Youden's J", "Default 0.5"],
        "Threshold Value": [f"{final_f1_threshold:.4f}", f"{final_youden_threshold:.4f}", "0.500"],
        "Accuracy": [ci_to_string(m['Accuracy']) for m in [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "ROC-AUC": [ci_to_string(m['ROC-AUC']) for m in [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "PR-AUC": [ci_to_string(m['PR-AUC']) for m in [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "Sensitivity (Recall, Positive)": [ci_to_string(m['Recall (Positive)']) for m in
                                           [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "Specificity (Recall, Negative)": [ci_to_string(m['Recall (Negative)']) for m in
                                           [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "Precision (Positive)": [ci_to_string(m['Precision (Positive)']) for m in
                                 [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "Precision (Negative)": [ci_to_string(m['Precision (Negative)']) for m in
                                 [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "F1 (Positive)": [ci_to_string(m['F1-Score (Positive)']) for m in
                          [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
        "F1 (Negative)": [ci_to_string(m['F1-Score (Negative)']) for m in
                          [metrics_f1_ci, metrics_youden_ci, metrics_05_ci]],
    }
    all_metrics_df = pd.DataFrame(report_data)

    excel_path = os.path.join(OUTPUT_DIR, "final_model_metrics_report.xlsx")
    all_metrics_df.to_excel(excel_path, index=False)

    print("\n🔍 Final Test Metrics Report (with 95% CI):")
    print(all_metrics_df.to_string())
    print(f"\n✅ Metrics report saved to Excel file: {excel_path}")

    # 5. YENİ GÖRSEL RAPORLARIN OLUŞTURULMASI
    print("\n--- 🖼️ Generating final plots ---")
    plot_combined_curves(y_true_cv, y_pred_prob_cv, "SYNAPSI_Ensemble")
    plot_calibration_curve(y_true_cv, y_pred_prob_cv, "SYNAPSI_Ensemble")

    thresholds_dict_combined_cm = {
        'A': final_youden_threshold,
        'B': 0.5
    }
    plot_combined_confusion_matrices(y_true_cv, y_pred_prob_cv, thresholds_dict_combined_cm,
                                     "combined_confusion_matrices.tiff")

    # 6. MODEL YORUMLANABİLİRLİĞİ (SHAP)
    print("\n--- 📊 Generating SHAP plots for interpretability ---")

    imputer_final = SimpleImputer(strategy='median')
    X_imputed_final = pd.DataFrame(imputer_final.fit_transform(features), columns=features.columns)

    final_catboost_shap = CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0)
    final_xgboost_shap = XGBClassifier(**best_params_xgboost, random_state=RANDOM_STATE)
    final_lightgbm_shap = LGBMClassifier(**best_params_lightgbm, random_state=RANDOM_STATE)

    final_catboost_shap.fit(X_imputed_final, target)
    final_xgboost_shap.fit(X_imputed_final, target)
    final_lightgbm_shap.fit(X_imputed_final, target)

    models_dict = {
        'CatBoost': final_catboost_shap,
        'XGBoost': final_xgboost_shap,
        'LightGBM': final_lightgbm_shap
    }

    for name, model in models_dict.items():
        plot_single_shap_summary(model, X_imputed_final, name, feature_display_names)

    # 7. DEPLOYMENT MODELİNİN KAYDEDİLMESİ
    deployment_model = VotingClassifier(
        estimators=[('CatBoost', CatBoostClassifier(**best_params_catboost, random_seed=RANDOM_STATE, verbose=0)),
                    ('XGBoost', XGBClassifier(**best_params_xgboost, random_state=RANDOM_STATE)),
                    ('LightGBM', LGBMClassifier(**best_params_lightgbm, random_state=RANDOM_STATE))],
        voting='soft',
        weights=dynamic_weights
    )
    deployment_model.fit(X_imputed_final, target)

    deployment_artifact = {
        'model': deployment_model, 'features': selected_features, 'imputer': imputer_final,
        'thresholds': {'f1_stable': final_f1_threshold, 'youden_stable': final_youden_threshold, 'default_0.5': 0.5},
        'hyperparameters': {'catboost': best_params_catboost, 'xgboost': best_params_xgboost,
                            'lightgbm': best_params_lightgbm},
        'model_weights': dynamic_weights
    }
    joblib.dump(deployment_artifact, os.path.join(OUTPUT_DIR, 'SYNAPSI_deployment_model.pkl'))
    print(f"\n✅ Deployment model and all artifacts saved to '{OUTPUT_DIR}' directory.")


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