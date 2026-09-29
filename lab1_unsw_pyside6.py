import sys
import os
import io
import re
import shutil
import tempfile
import traceback
import threading
from pathlib import Path

import requests
import pandas as pd
import numpy as np


import matplotlib.pyplot as plt
import seaborn as sns



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif, RFE
from sklearn.tree import DecisionTreeClassifier

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot, Qt, QUrl
from PySide6.QtGui import QDesktopServices, QCloseEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QLabel, QVBoxLayout, QHBoxLayout,
    QStackedLayout, QPlainTextEdit, QPushButton, QProgressBar, QMessageBox,
    QGraphicsDropShadowEffect,
)

import matplotlib
matplotlib.use("Agg")

RANDOM_STATE = 42

# Каждый запуск анализа заново скачивает CSV.
# Hugging Face — основной источник, GitHub — резервный.
DATASET_URLS = [
    "https://huggingface.co/datasets/Mouwiya/UNSW-NB15/resolve/main/UNSW_NB15_training-set.csv",
    "https://raw.githubusercontent.com/Nir-J/ML-Projects/master/UNSW-Network_Packet_Classification/UNSW_NB15_training-set.csv",
]

SOURCE_FILENAME = "UNSW_NB15_training-set.csv"


class CancelledError(Exception):
    pass


class AnalysisSignals(QObject):
    progress = Signal(int, str)
    log = Signal(str)
    result = Signal(object)
    error = Signal(str)
    finished = Signal()


class UNSWAnalysisTask(QRunnable):
    """Полный конвейер лабораторной работы №1."""

    def __init__(self, work_dir: str):
        super().__init__()
        self.signals = AnalysisSignals()
        self.work_dir = Path(work_dir)
        self.cancel_event = threading.Event()
        self.source_csv = self.work_dir / SOURCE_FILENAME

    def cancel(self):
        self.cancel_event.set()

    def check_cancelled(self):
        if self.cancel_event.is_set():
            raise CancelledError("Вычисления отменены пользователем.")

    def emit_progress(self, value: int, text: str):
        self.signals.progress.emit(value, text)
        self.signals.log.emit(f"\n[{value:3d}%] {text}")

    def log(self, text=""):
        self.signals.log.emit(str(text))

    def save_fig(self, filename: str):
        path = self.work_dir / filename
        plt.tight_layout()
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        self.log(f"   Сохранён рисунок: {path.name}")

    def download_dataset(self):
        self.emit_progress(5, "Загрузка актуальной опубликованной копии UNSW-NB15")
        last_error = None
        min_size = 5_000_000

        for url in DATASET_URLS:
            self.check_cancelled()
            self.log(f"Источник: {url}")
            tmp_path = self.source_csv.with_suffix(".part")

            try:
                with requests.get(
                    url,
                    stream=True,
                    timeout=(15, 180),
                    headers={"User-Agent": "Lab1-UNSW-NB15/1.0"},
                ) as response:
                    self.log(f"HTTP status: {response.status_code}")
                    response.raise_for_status()

                    total = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(tmp_path, "wb") as f:
                        for chunk in response.iter_content(chunk_size=1024 * 1024):
                            self.check_cancelled()
                            if not chunk:
                                continue
                            f.write(chunk)
                            downloaded += len(chunk)

                            if total:
                                pct = 5 + int(min(downloaded / total, 1.0) * 10)
                                self.signals.progress.emit(
                                    pct,
                                    f"Скачивание: {downloaded / 1024**2:.1f} / "
                                    f"{total / 1024**2:.1f} MB",
                                )

                size = tmp_path.stat().st_size
                self.log(f"Получено: {size / 1024**2:.2f} MB")

                if size < min_size:
                    raise ValueError(
                        f"Файл подозрительно мал: {size:,} байт "
                        f"(ожидалось > {min_size:,})."
                    )

                os.replace(tmp_path, self.source_csv)
                self.log(f"CSV создан: {self.source_csv}")
                return

            except Exception as exc:
                last_error = exc
                self.log(f"Источник недоступен: {exc}")
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

        raise RuntimeError(
            "Не удалось скачать UNSW-NB15 ни из одного источника. "
            f"Последняя ошибка: {last_error}"
        )

    @Slot()
    def run(self):
        np.random.seed(RANDOM_STATE)
        try:
            self.work_dir.mkdir(parents=True, exist_ok=True)
            self.download_dataset()
            self.check_cancelled()

            # ============================================================
            # 1. Чтение и проверка
            # ============================================================
            self.emit_progress(18, "Пункт 1. Чтение и проверка данных")
            df = pd.read_csv(self.source_csv)

            required = {
                "id", "proto", "service", "state", "attack_cat", "label",
                "dur", "sbytes", "dbytes", "spkts", "dpkts",
                "sttl", "dttl", "sload", "dload",
            }
            missing_columns = sorted(required - set(df.columns))
            if missing_columns:
                raise ValueError(
                    "Файл прочитан, но не похож на ожидаемый training-set. "
                    f"Нет столбцов: {missing_columns}"
                )

            self.log(f"Размер набора: {df.shape[0]:,} строк × {df.shape[1]} столбцов")
            self.log("\nПервые 5 строк:\n" + df.head().to_string())

            info_buf = io.StringIO()
            df.info(buf=info_buf)
            self.log("\n.info():\n" + info_buf.getvalue())
            self.log("\nТипы данных:\n" + df.dtypes.value_counts().to_string())
            self.log("\n.describe().T (первые 15 строк):\n" + df.describe().T.head(15).to_string())
            self.log(f"\nДубликатов строк: {df.duplicated().sum():,}")
            self.log(f"id уникален: {df['id'].is_unique}")

            cat_cols = df.select_dtypes(include="object").columns.tolist()
            self.log(f"Категориальные столбцы: {cat_cols}")
            self.check_cancelled()

            # ============================================================
            # 2. Пропуски
            # ============================================================
            self.emit_progress(26, "Пункт 2. Проверка и обработка пропусков")
            self.log(f"Явных NaN до обработки: {int(df.isna().sum().sum()):,}")

            for col in cat_cols:
                top = df[col].value_counts(dropna=False).head(5).to_dict()
                self.log(f"{col}: {df[col].nunique(dropna=False)} значений -> {top}")

            n_dash = int((df["service"] == "-").sum())
            self.log(
                f"Скрытых пропусков '-' в service: {n_dash:,} "
                f"({n_dash / len(df):.2%})"
            )

            df["service"] = df["service"].replace("-", np.nan)

            # Визуализируем не более 5000 строк, статистика считается по всему набору.
            miss_sample = df.head(min(5000, len(df)))
            plt.figure(figsize=(12, 5))
            sns.heatmap(miss_sample.isna(), cbar=False, yticklabels=False, cmap="viridis")
            plt.title("Рис. 1. Тепловая карта пропусков (первые 5000 строк)")
            self.save_fig("fig01_missing_heatmap.png")

            self.log(
                f"Доля атак при service=NaN: "
                f"{df.loc[df['service'].isna(), 'label'].mean():.3f}"
            )
            self.log(
                f"Доля атак при известном service: "
                f"{df.loc[df['service'].notna(), 'label'].mean():.3f}"
            )

            df["service"] = df["service"].fillna("unknown")
            self.log(f"NaN после обработки: {int(df.isna().sum().sum()):,}")
            self.check_cancelled()

            # ============================================================
            # 3. Структура + regex
            # ============================================================
            self.emit_progress(34, "Пункт 3. Структура набора и regex")
            df = df.drop(columns=["id"])

            df["flow_desc"] = (
                df["proto"].astype(str) + "|" +
                df["service"].astype(str) + "|" +
                df["state"].astype(str)
            )

            parsed = df["flow_desc"].str.extract(
                r"^(?P<p_proto>[^|]+)\|(?P<p_service>[^|]+)\|(?P<p_state>[^|]+)$"
            )

            self.log(
                "Совпадение после regex: "
                f"proto={(parsed['p_proto'] == df['proto']).mean():.3f}, "
                f"service={(parsed['p_service'] == df['service']).mean():.3f}, "
                f"state={(parsed['p_state'] == df['state']).mean():.3f}"
            )
            df = df.drop(columns=["flow_desc"])

            def clean_category(value):
                value = str(value).strip().lower()
                return re.sub(r"\s+", "_", value)

            df["attack_cat"] = df["attack_cat"].apply(clean_category)
            df["is_tcp"] = df["proto"].str.contains(r"^tcp$", regex=True, na=False).astype(int)

            top_proto = df["proto"].value_counts().head(5).index.tolist()
            self.log("Топ-5 proto: " + str(df["proto"].value_counts().head(5).to_dict()))
            df["proto_grp"] = df["proto"].where(df["proto"].isin(top_proto), "other")
            self.check_cancelled()

            # ============================================================
            # 4. Кодирование
            # ============================================================
            self.emit_progress(41, "Пункт 4. Кодирование категориальных признаков")

            le_state = LabelEncoder()
            df["state_le"] = le_state.fit_transform(df["state"])
            self.log(
                "LabelEncoder(state): " +
                str(dict(zip(le_state.classes_, le_state.transform(le_state.classes_))))
            )

            le_cat = LabelEncoder()
            df["attack_cat_le"] = le_cat.fit_transform(df["attack_cat"])
            self.log(
                "LabelEncoder(attack_cat): " +
                str(dict(zip(le_cat.classes_, le_cat.transform(le_cat.classes_))))
            )

            freq = df["state"].value_counts(normalize=True)
            df["state_freq"] = df["state"].map(freq)
            self.log("Frequency Encoding(state): " + str(freq.round(3).to_dict()))

            df = pd.get_dummies(
                df,
                columns=["proto_grp", "service"],
                prefix=["proto", "svc"],
                dtype=int,
            )
            self.log(f"Размер после One-Hot: {df.shape}")
            self.check_cancelled()

            # ============================================================
            # 5. Визуализация
            # ============================================================
            self.emit_progress(48, "Пункт 5. Визуализация данных")

            order = df["attack_cat"].value_counts().index
            plt.figure(figsize=(10, 5))
            sns.countplot(data=df, x="attack_cat", order=order)
            plt.xticks(rotation=45, ha="right")
            plt.title("Рис. 2. Распределение категорий атак")
            self.save_fig("fig02_attack_cat_count.png")

            num_show = [
                "dur", "sbytes", "dbytes", "spkts", "dpkts",
                "rate", "sload", "dload", "smean",
            ]
            fig, axes = plt.subplots(3, 3, figsize=(14, 10))
            for ax, col in zip(axes.flat, num_show):
                ax.hist(np.log1p(df[col].clip(lower=0)), bins=50)
                ax.set_title(f"log1p({col})")
            fig.suptitle("Рис. 3. Гистограммы числовых признаков")
            self.save_fig("fig03_histograms.png")

            num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if "attack_cat_le" in num_cols:
                num_cols.remove("attack_cat_le")
            corr = df[num_cols].corr()

            plt.figure(figsize=(18, 15))
            sns.heatmap(corr, cmap="coolwarm", center=0, square=False)
            plt.title("Рис. 4. Матрица корреляций")
            self.save_fig("fig04_corr_matrix.png")

            high = (
                corr.abs()
                .where(np.triu(np.ones(corr.shape), k=1).astype(bool))
                .stack()
                .sort_values(ascending=False)
            )
            self.log("\nПары с |r| > 0.95:\n" + high[high > 0.95].head(30).round(3).to_string())

            if "label" in corr.columns:
                self.log(
                    "\nНаибольшая |корреляция| с label:\n" +
                    corr["label"].abs().sort_values(ascending=False).head(10).round(3).to_string()
                )

            plt.figure(figsize=(12, 5))
            sns.boxplot(
                data=df,
                x="attack_cat",
                y=np.log1p(df["sbytes"].clip(lower=0)),
                order=order,
            )
            plt.xticks(rotation=45, ha="right")
            plt.ylabel("log1p(sbytes)")
            plt.title("Рис. 5. sbytes по типам атак")
            self.save_fig("fig05_boxplot_sbytes.png")

            proto_cols = [c for c in df.columns if c.startswith("proto_")]
            proto_share = pd.DataFrame({
                c.replace("proto_", ""):
                    df.loc[df[c] == 1, "label"].value_counts(normalize=True)
                for c in proto_cols
            }).T.fillna(0)
            proto_share = proto_share.reindex(columns=[0, 1], fill_value=0)
            proto_share.columns = ["normal", "attack"]
            proto_share.plot(kind="bar", stacked=True, figsize=(9, 5))
            plt.title("Рис. 6. Доля атак по группам протоколов")
            plt.ylabel("Доля")
            self.save_fig("fig06_proto_stacked.png")
            self.log("\nДоля атак по протоколам:\n" + proto_share.round(3).to_string())

            svc_cols = [c for c in df.columns if c.startswith("svc_")]
            svc_attack = pd.DataFrame({
                c.replace("svc_", ""):
                    df.loc[df[c] == 1, "attack_cat"].value_counts()
                for c in svc_cols
            }).fillna(0)
            plt.figure(figsize=(12, 7))
            sns.heatmap(
                np.log1p(svc_attack),
                cmap="YlOrRd",
                annot=svc_attack.astype(int),
                fmt="d",
            )
            plt.title("Рис. 7. Сервис × тип атаки")
            self.save_fig("fig07_service_attack_heatmap.png")

            sample_n = min(20_000, len(df))
            sample = df.sample(sample_n, random_state=RANDOM_STATE)
            plt.figure(figsize=(8, 6))
            plt.scatter(
                np.log1p(sample["sbytes"].clip(lower=0)),
                np.log1p(sample["dbytes"].clip(lower=0)),
                c=sample["label"],
                cmap="coolwarm",
                s=4,
                alpha=0.5,
            )
            plt.xlabel("log1p(sbytes)")
            plt.ylabel("log1p(dbytes)")
            plt.title("Рис. 8. sbytes–dbytes по label")
            self.save_fig("fig08_scatter.png")

            plt.figure(figsize=(8, 5))
            for lbl, name in [(0, "норма"), (1, "атака")]:
                plt.hist(
                    df.loc[df["label"] == lbl, "sttl"],
                    bins=60,
                    alpha=0.6,
                    label=name,
                )
            plt.legend()
            plt.xlabel("sttl")
            plt.title("Рис. 9. Распределение sttl по классам")
            self.save_fig("fig09_sttl_hist.png")

            self.log(
                "sttl ∈ {254,255}: " +
                str(df.loc[df["sttl"].isin([254, 255]), "label"].value_counts().to_dict())
            )
            self.check_cancelled()

            # ============================================================
            # 6. Feature engineering + отбор
            # ============================================================
            self.emit_progress(62, "Пункт 6. Feature engineering и отбор признаков")

            df["bytes_ratio"] = df["sbytes"] / (df["dbytes"] + 1)
            df["pkts_total"] = df["spkts"] + df["dpkts"]
            df["bytes_per_pkt"] = (
                (df["sbytes"] + df["dbytes"]) / (df["pkts_total"] + 1)
            )
            df["ttl_diff"] = (df["sttl"] - df["dttl"]).abs()

            drop_cols = ["proto", "state", "attack_cat", "attack_cat_le", "label"]
            X_all = df.drop(columns=drop_cols)
            y = df["label"].astype(int)

            non_numeric = X_all.select_dtypes(exclude=[np.number]).columns.tolist()
            if non_numeric:
                raise ValueError(
                    "После кодирования остались строковые признаки: " + ", ".join(non_numeric)
                )

            self.log(f"Признаков до отбора: {X_all.shape[1]}")

            skew = X_all.skew(numeric_only=True).abs()
            log_cols = skew[skew > 3].index.tolist()
            log_cols = [
                c for c in log_cols
                if X_all[c].min() >= 0 and X_all[c].nunique() > 2
            ]
            for col in log_cols:
                X_all[col] = np.log1p(
                X_all[col].astype("float64")
                )
            self.log(f"Логарифмировано {len(log_cols)} признаков: {log_cols}")

            # Порядок оставлен как в предоставленном отчёте.
            X_mm = pd.DataFrame(
                MinMaxScaler().fit_transform(X_all),
                columns=X_all.columns,
                index=X_all.index,
            )
            vt = VarianceThreshold(threshold=0.01).fit(X_mm)
            low_var = X_all.columns[~vt.get_support()].tolist()
            self.log(f"Удалено по низкой дисперсии ({len(low_var)}): {low_var}")

            X_sel = X_all.drop(columns=low_var)

            corr_m = X_sel.corr().abs()
            upper = corr_m.where(np.triu(np.ones(corr_m.shape), k=1).astype(bool))
            multicol = [c for c in upper.columns if (upper[c] > 0.95).any()]
            self.log(f"Удалено мультиколлинеарных ({len(multicol)}): {multicol}")
            X_sel = X_sel.drop(columns=multicol)
            self.log(f"Осталось после фильтров: {X_sel.shape[1]}")
            self.check_cancelled()

            k = min(20, X_sel.shape[1])
            skb = SelectKBest(f_classif, k=k).fit(X_sel, y)
            scores = pd.Series(skb.scores_, index=X_sel.columns)
            scores = scores.replace([np.inf, -np.inf], np.nan).fillna(0).sort_values(ascending=False)
            kbest_feats = scores.head(k).index.tolist()
            self.log("\nТоп-10 SelectKBest:\n" + scores.head(10).round(1).to_string())

            plt.figure(figsize=(9, 7))
            scores.head(k).sort_values().plot(kind="barh")
            plt.title(f"Рис. 10. SelectKBest: {k} лучших признаков")
            self.save_fig("fig10_selectkbest.png")
            self.check_cancelled()

            self.emit_progress(72, "RFE: рекурсивный отбор признаков")
            sample_size = min(30_000, len(X_sel))
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(len(X_sel), sample_size, replace=False)

            n_rfe = min(15, X_sel.shape[1])
            rfe = RFE(
                DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
                n_features_to_select=n_rfe,
                step=3,
            )
            rfe.fit(X_sel.iloc[idx], y.iloc[idx])
            self.check_cancelled()

            rfe_feats = X_sel.columns[rfe.support_].tolist()
            self.log(f"RFE выбрал ({len(rfe_feats)}): {rfe_feats}")

            final_feats = sorted(set(kbest_feats) | set(rfe_feats))
            self.log(f"Итоговых признаков: {len(final_feats)}")
            self.log(str(final_feats))
            X = X_sel[final_feats].copy()

            # ============================================================
            # 7. Баланс классов
            # ============================================================
            self.emit_progress(80, "Пункт 7. Проверка сбалансированности классов")

            label_counts = y.value_counts().sort_index()
            label_share = y.value_counts(normalize=True).sort_index()
            self.log("\nlabel counts:\n" + label_counts.to_string())
            self.log("\nlabel share:\n" + label_share.round(4).to_string())
            self.log(
                "\nattack_cat share:\n" +
                df["attack_cat"].value_counts(normalize=True).round(4).to_string()
            )

            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            label_counts.plot(
                kind="pie",
                autopct="%.1f%%",
                labels=[f"class {i}" for i in label_counts.index],
                ax=axes[0],
            )
            axes[0].set_ylabel("")
            axes[0].set_title("label")

            df["attack_cat"].value_counts().plot(kind="bar", ax=axes[1], logy=True)
            axes[1].set_title("attack_cat (log-шкала)")
            fig.suptitle("Рис. 11. Баланс классов")
            self.save_fig("fig11_class_balance.png")
            self.check_cancelled()

            # ============================================================
            # 8. X/y + train/test
            # ============================================================
            self.emit_progress(86, "Пункт 8. Разбиение X/y и train/test")

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=0.3,
                random_state=RANDOM_STATE,
                stratify=y,
            )

            self.log(f"train: {X_train.shape}, test: {X_test.shape}")
            self.log(
                f"Доля атак train={y_train.mean():.4f}, test={y_test.mean():.4f}"
            )
            self.check_cancelled()

            # ============================================================
            # 9. StandardScaler
            # ============================================================
            self.emit_progress(92, "Пункт 9. Нормализация StandardScaler")
            self.log(
                "\nДиапазоны до масштабирования:\n" +
                X_train.agg(["min", "max"]).T.head(10).to_string()
            )

            scaler = StandardScaler()
            X_train_s = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=final_feats,
                index=X_train.index,
            )
            X_test_s = pd.DataFrame(
                scaler.transform(X_test),
                columns=final_feats,
                index=X_test.index,
            )

            self.log(f"train: max |mean| = {X_train_s.mean().abs().max():.5f}")
            self.log(
                "train: std (диапазон) = "
                f"{X_train_s.std().min():.3f} ... {X_train_s.std().max():.3f}"
            )
            self.log(
                "test mean (диапазон) = "
                f"{X_test_s.mean().min():.3f} ... {X_test_s.mean().max():.3f}"
            )

            train_out = self.work_dir / "unsw_train_prepared.csv"
            test_out = self.work_dir / "unsw_test_prepared.csv"
            X_train_s.assign(label=y_train).to_csv(train_out, index=False)
            X_test_s.assign(label=y_test).to_csv(test_out, index=False)

            self.check_cancelled()
            self.emit_progress(100, "Готово")

            self.signals.result.emit({
                "source_csv": str(self.source_csv),
                "train_csv": str(train_out),
                "test_csv": str(test_out),
                "work_dir": str(self.work_dir),
                "rows": len(df),
                "final_features": final_feats,
                "train_shape": X_train_s.shape,
                "test_shape": X_test_s.shape,
                "attack_share_train": float(y_train.mean()),
                "attack_share_test": float(y_test.mean()),
            })

        except CancelledError as exc:
            self.signals.log.emit(f"\n⛔ {exc}")
        except Exception:
            self.signals.error.emit(traceback.format_exc())
        finally:
            plt.close("all")
            self.signals.finished.emit()


class GlowButton(QPushButton):
    def __init__(self, text: str):
        super().__init__(text)
        self.shadow = QGraphicsDropShadowEffect(self)
        self.shadow.setBlurRadius(15)
        self.shadow.setOffset(2, 2)
        self.setGraphicsEffect(self.shadow)

    def enterEvent(self, event):
        self.shadow.setBlurRadius(35)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.shadow.setBlurRadius(15)
        super().leaveEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Лабораторная работа №1 — UNSW-NB15")
        self.resize(980, 760)

        # Временная папка живёт только пока открыто приложение.
        self.temp_dir = tempfile.mkdtemp(prefix="lab1_unsw_")

        self.thread_pool = QThreadPool.globalInstance()
        self.current_task = None
        self.is_running = False
        self.close_requested = False
        self.last_result = None

        self.setup_ui()
        self.setup_signals()
        self.apply_styles()

        self.statusBar().showMessage(
            "Готово. CSV будет скачан заново после нажатия Start."
        )

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        self.stacked = QStackedLayout()
        main_layout.addLayout(self.stacked)

        # ---------------- Page 1 ----------------
        self.page_task = QWidget()
        task_layout = QVBoxLayout(self.page_task)

        title = QLabel("Лабораторная работа №1")
        title.setObjectName("task_title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        task_layout.addWidget(title)

        html = """
        <h2>Предобработка UNSW-NB15 для обнаружения сетевых вторжений</h2>
        <p><b>Дисциплина:</b> Модели и методы машинного обучения</p>
        <p><b>Преподаватель:</b> Вовик А.Г.</p>
        <p><b>Студентка:</b> Дарбека Елена, МФБ2631</p>
        <hr>
        <p>Приложение выполняет все 9 пунктов лабораторной:</p>
        <ol>
          <li>Pandas: загрузка и исследование данных;</li>
          <li>поиск и обработка скрытых пропусков;</li>
          <li>изменение структуры и регулярные выражения;</li>
          <li>Label / One-Hot / Frequency Encoding;</li>
          <li>визуализация;</li>
          <li>feature engineering и отбор признаков;</li>
          <li>анализ баланса классов;</li>
          <li>X/y и train/test;</li>
          <li>StandardScaler.</li>
        </ol>
        <p><b>Важно:</b> при каждом запуске анализа исходный CSV скачивается заново.
        Все CSV и PNG помещаются во временный каталог и удаляются при закрытии приложения.</p>
        """
        description = QLabel(html)
        description.setObjectName("task_description")
        description.setWordWrap(True)
        description.setTextFormat(Qt.TextFormat.RichText)
        description.setAlignment(Qt.AlignmentFlag.AlignCenter)
        task_layout.addWidget(description)
        task_layout.addStretch()

        self.button_continue = GlowButton("Продолжить  ✅")
        task_layout.addWidget(self.button_continue)

        # ---------------- Page 2 ----------------
        self.page_calc = QWidget()
        calc_layout = QVBoxLayout(self.page_calc)

        calc_title = QLabel("Выполнение лабораторной работы")
        calc_title.setObjectName("task_title")
        calc_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        calc_layout.addWidget(calc_title)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setFormat("%p% — готово")
        calc_layout.addWidget(self.progress)

        self.result_box = QPlainTextEdit()
        self.result_box.setReadOnly(True)
        calc_layout.addWidget(self.result_box, 1)

        buttons = QHBoxLayout()
        self.button_start = GlowButton("Start  ✅")
        buttons.addWidget(self.button_start)

        self.button_cancel = GlowButton("Cancel  ❌")
        self.button_cancel.setEnabled(False)
        buttons.addWidget(self.button_cancel)

        self.button_folder = GlowButton("Открыть временные файлы")
        self.button_folder.setEnabled(False)
        buttons.addWidget(self.button_folder)

        self.button_back = GlowButton("⬅  Back")
        buttons.addWidget(self.button_back)
        calc_layout.addLayout(buttons)

        self.stacked.addWidget(self.page_task)
        self.stacked.addWidget(self.page_calc)
        self.stacked.setCurrentWidget(self.page_task)

    def setup_signals(self):
        self.button_continue.clicked.connect(
            lambda: self.stacked.setCurrentWidget(self.page_calc)
        )
        self.button_back.clicked.connect(self.switch_to_task)
        self.button_start.clicked.connect(self.start_analysis)
        self.button_cancel.clicked.connect(self.cancel_analysis)
        self.button_folder.clicked.connect(self.open_temp_folder)

    def switch_to_task(self):
        if self.is_running:
            self.result_box.appendPlainText(
                "\n⚠️ Сначала дождитесь завершения или нажмите Cancel."
            )
            return
        self.stacked.setCurrentWidget(self.page_task)

    def start_analysis(self):
        if self.is_running:
            return

        # Новый запуск = новая копия данных и новый временный каталог.
        self.cleanup_temp_dir()
        self.temp_dir = tempfile.mkdtemp(prefix="lab1_unsw_")

        self.result_box.clear()
        self.progress.setValue(0)
        self.result_box.appendPlainText(
            "🚀 Запуск лабораторной работы UNSW-NB15\n"
            "Исходный CSV будет скачан заново.\n" + "-" * 72
        )

        self.is_running = True
        self.last_result = None
        self.button_start.setEnabled(False)
        self.button_cancel.setEnabled(True)
        self.button_back.setEnabled(False)
        self.button_folder.setEnabled(False)

        self.current_task = UNSWAnalysisTask(self.temp_dir)
        self.current_task.signals.progress.connect(self.on_progress)
        self.current_task.signals.log.connect(self.on_log)
        self.current_task.signals.result.connect(self.on_result)
        self.current_task.signals.error.connect(self.on_error)
        self.current_task.signals.finished.connect(self.on_finished)

        self.thread_pool.start(self.current_task)

    def cancel_analysis(self):
        if self.current_task and self.is_running:
            self.current_task.cancel()
            self.button_cancel.setEnabled(False)
            self.statusBar().showMessage(
                "Отмена запрошена. Текущий sklearn.fit может завершиться не сразу."
            )
            self.result_box.appendPlainText("\n⏳ Запрошена отмена...")

    @Slot(int, str)
    def on_progress(self, value, text):
        self.progress.setValue(value)
        self.progress.setFormat(f"%p% — {text}")
        self.statusBar().showMessage(text)

    @Slot(str)
    def on_log(self, text):
        self.result_box.appendPlainText(text)
        scrollbar = self.result_box.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    @Slot(object)
    def on_result(self, result):
        self.last_result = result
        self.button_folder.setEnabled(True)

        self.result_box.appendPlainText("\n" + "=" * 72)
        self.result_box.appendPlainText("✅ ЛАБОРАТОРНАЯ РАБОТА ЗАВЕРШЕНА")
        self.result_box.appendPlainText(f"Исходный CSV: {result['source_csv']}")
        self.result_box.appendPlainText(f"Prepared train: {result['train_csv']}")
        self.result_box.appendPlainText(f"Prepared test: {result['test_csv']}")
        self.result_box.appendPlainText(
            f"Итоговых признаков: {len(result['final_features'])}"
        )
        self.result_box.appendPlainText(
            f"Train: {result['train_shape']}; Test: {result['test_shape']}"
        )
        self.result_box.appendPlainText(
            "\nВсе эти файлы временные и будут удалены при закрытии приложения."
        )

    @Slot(str)
    def on_error(self, error_text):
        self.result_box.appendPlainText("\n❌ ОШИБКА:\n" + error_text)
        QMessageBox.critical(
            self,
            "Ошибка",
            "Во время обработки произошла ошибка.\nПодробности выведены в журнал.",
        )

    @Slot()
    def on_finished(self):
        self.is_running = False
        self.current_task = None
        self.button_start.setEnabled(True)
        self.button_cancel.setEnabled(False)
        self.button_back.setEnabled(True)

        if self.last_result:
            self.progress.setValue(100)
            self.progress.setFormat("100% — завершено")
            self.statusBar().showMessage("Работа завершена.")
        else:
            self.statusBar().showMessage("Вычисления остановлены.")

        if self.close_requested:
            self.close()

    def open_temp_folder(self):
        if os.path.isdir(self.temp_dir):
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.temp_dir))

    def cleanup_temp_dir(self):
        if getattr(self, "temp_dir", None):
            shutil.rmtree(self.temp_dir, ignore_errors=True)

    def closeEvent(self, event: QCloseEvent):
        if self.is_running:
            answer = QMessageBox.question(
                self,
                "Вычисления выполняются",
                "Сейчас выполняется анализ.\n\n"
                "Запросить отмену и закрыть приложение после безопасного "
                "завершения текущего этапа?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if answer == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.close_requested = True
            if self.current_task:
                self.current_task.cancel()
            self.hide()
            event.ignore()
            return

        self.cleanup_temp_dir()
        event.accept()

    def apply_styles(self):
        self.setStyleSheet("""
            QWidget {
                font-family: Arial;
                font-size: 13pt;
            }
            QLabel#task_title {
                font-size: 22pt;
                font-weight: bold;
                color: #2c3e50;
            }
            QLabel#task_description {
                font-size: 12pt;
                color: #333333;
            }
            QPlainTextEdit {
                font-family: Menlo, Consolas, monospace;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #6f9f72;
                color: white;
                border: none;
                border-radius: 7px;
                padding: 10px 14px;
                font-size: 12pt;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #82b584; }
            QPushButton:pressed { background-color: #4f7652; }
            QPushButton:disabled {
                background-color: #9b9b9b;
                color: #e5e5e5;
            }
            QProgressBar {
                min-height: 28px;
                text-align: center;
                font-weight: bold;
            }
        """)


def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
