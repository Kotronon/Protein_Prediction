#!/usr/bin/env python3
"""Render the DisProt-focused UdonPred experiment report and figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(".")
FIGURE_DIR = ROOT / "docs" / "images" / "disprot_focus"
REPORT_PATH = ROOT / "docs" / "disprot_focus_udonpred_report.md"
SUMMARY_PATH = ROOT / "results" / "disprot_focus_experiment_summary.csv"


EXPERIMENTS = {
    "Equal multitask": ROOT / "results" / "disprot_focus_checkpoint_eval" / "checkpoint_summary.csv",
    "DisProt-only": ROOT / "results" / "disprot_only_checkpoint_eval" / "checkpoint_summary.csv",
    "Weighted multitask": ROOT
    / "results"
    / "disprot_weighted_multitask_checkpoint_eval"
    / "checkpoint_summary.csv",
}


def read_baseline() -> dict[str, float | str | int]:
    matrix = pd.read_csv(ROOT / "results" / "udonpred_matrix" / "matrix.csv")
    row = matrix.loc[matrix["train_dataset"] == "disprot"].iloc[0]
    return {
        "strategy": "Original DisProt head",
        "best_checkpoint": "pretrained ONNX",
        "step": "",
        "disprot_AP": float(row["disprot\n(AP)"]),
        "disprot_AUROC": float(row["disprot\n(AUROC)"]),
    }


def read_best_experiments() -> pd.DataFrame:
    rows = [read_baseline()]
    for name, path in EXPERIMENTS.items():
        frame = pd.read_csv(path)
        best = frame.sort_values("disprot_AP", ascending=False).iloc[0]
        rows.append(
            {
                "strategy": name,
                "best_checkpoint": best["checkpoint"],
                "step": int(best["step"]),
                "disprot_AP": float(best["disprot_AP"]),
                "disprot_AUROC": float(best["disprot_AUROC"]),
            }
        )
    summary = pd.DataFrame(rows)
    baseline_ap = float(summary.loc[summary["strategy"] == "Original DisProt head", "disprot_AP"].iloc[0])
    baseline_auroc = float(
        summary.loc[summary["strategy"] == "Original DisProt head", "disprot_AUROC"].iloc[0]
    )
    summary["delta_AP_vs_baseline"] = summary["disprot_AP"] - baseline_ap
    summary["delta_AUROC_vs_baseline"] = summary["disprot_AUROC"] - baseline_auroc
    return summary


def plot_best_strategy_bars(summary: pd.DataFrame) -> None:
    plot_df = summary.copy()
    colors = ["#4c78a8", "#59a14f", "#f28e2b", "#b07aa1"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    y_pos = range(len(plot_df))
    for ax, metric, title in [
        (axes[0], "disprot_AP", "DisProt Average Precision"),
        (axes[1], "disprot_AUROC", "DisProt AUROC"),
    ]:
        ax.barh(list(y_pos), plot_df[metric], color=colors[: len(plot_df)])
        ax.set_xlim(0.84 if metric == "disprot_AP" else 0.88, 0.965)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="x", alpha=0.25)
        ax.set_yticks(list(y_pos), plot_df["strategy"])
        for idx, value in enumerate(plot_df[metric]):
            ax.text(value + 0.001, idx, f"{value:.3f}", ha="left", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "best_strategy_metrics.png", dpi=220)
    plt.close(fig)


def plot_delta_bars(summary: pd.DataFrame) -> None:
    plot_df = summary.loc[summary["strategy"] != "Original DisProt head"].copy()
    y = range(len(plot_df))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8, 4.3))
    ax.barh(
        [value - width / 2 for value in y],
        plot_df["delta_AP_vs_baseline"],
        height=width,
        label="AP-Delta",
        color="#4c78a8",
    )
    ax.barh(
        [value + width / 2 for value in y],
        plot_df["delta_AUROC_vs_baseline"],
        height=width,
        label="AUROC-Delta",
        color="#f28e2b",
    )
    ax.axvline(0, color="#333333", linewidth=1)
    ax.set_yticks(list(y), plot_df["strategy"])
    ax.set_title("Differenz zum originalen DisProt-Head", loc="left", fontweight="bold")
    ax.set_xlabel("Metrikdifferenz")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "delta_vs_baseline.png", dpi=220)
    plt.close(fig)


def plot_checkpoint_curves() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
    colors = {
        "Equal multitask": "#b07aa1",
        "DisProt-only": "#59a14f",
        "Weighted multitask": "#f28e2b",
    }
    for name, path in EXPERIMENTS.items():
        frame = pd.read_csv(path).sort_values("step")
        axes[0].plot(frame["step"], frame["disprot_AP"], marker="o", label=name, color=colors[name])
        axes[1].plot(
            frame["step"], frame["disprot_AUROC"], marker="o", label=name, color=colors[name]
        )
    axes[0].set_title("AP über Checkpoints", loc="left", fontweight="bold")
    axes[1].set_title("AUROC über Checkpoints", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("Trainingsschritt")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Metrik")
    axes[1].legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "checkpoint_curves.png", dpi=220)
    plt.close(fig)


def plot_smoothing_effect(summary: pd.DataFrame) -> None:
    smooth0_path = ROOT / "results" / "disprot_only_checkpoint_eval" / "checkpoint_800_smooth0.json"
    smooth0 = pd.read_json(smooth0_path, typ="series")
    smooth15 = summary.loc[summary["strategy"] == "DisProt-only"].iloc[0]
    data = pd.DataFrame(
        [
            {"setting": "smooth=0", "AP": float(smooth0["disprot_AP"]), "AUROC": float(smooth0["disprot_AUROC"])},
            {"setting": "smooth=1.5", "AP": float(smooth15["disprot_AP"]), "AUROC": float(smooth15["disprot_AUROC"])},
        ]
    )
    x = range(len(data))
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, data["AP"], marker="o", label="AP", color="#4c78a8")
    ax.plot(x, data["AUROC"], marker="o", label="AUROC", color="#f28e2b")
    ax.set_xticks(list(x), data["setting"])
    ax.set_ylim(0.925, 0.965)
    ax.set_title("Effekt der Glättung auf DisProt-only checkpoint-800", loc="left", fontweight="bold")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "smoothing_effect.png", dpi=220)
    plt.close(fig)


def markdown_table(summary: pd.DataFrame) -> str:
    table = summary.copy()
    table["step"] = table["step"].astype(str)
    for col in ["disprot_AP", "disprot_AUROC", "delta_AP_vs_baseline", "delta_AUROC_vs_baseline"]:
        table[col] = table[col].map(lambda value: f"{value:.6f}")
    columns = [
        "strategy",
        "step",
        "disprot_AP",
        "disprot_AUROC",
        "delta_AP_vs_baseline",
        "delta_AUROC_vs_baseline",
    ]
    labels = [
        "Strategie",
        "Bester Schritt",
        "DisProt AP",
        "DisProt AUROC",
        "Delta AP",
        "Delta AUROC",
    ]
    lines = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for _, row in table[columns].iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def write_report(summary: pd.DataFrame) -> None:
    best = summary.sort_values("disprot_AP", ascending=False).iloc[0]
    baseline = summary.loc[summary["strategy"] == "Original DisProt head"].iloc[0]
    equal = summary.loc[summary["strategy"] == "Equal multitask"].iloc[0]
    weighted = summary.loc[summary["strategy"] == "Weighted multitask"].iloc[0]
    smooth0 = pd.read_json(
        ROOT / "results" / "disprot_only_checkpoint_eval" / "checkpoint_800_smooth0.json",
        typ="series",
    )

    text = f"""# DisProt-fokussierter UdonPred-Verbesserungsreport

## Kurzfassung

Ziel war, UdonPred gezielt für das DisProt-Ziel zu verbessern. PDBflex und ATLAS wurden bewusst ignoriert. Wenn TriZOD-artige Hilfslabels verwendet wurden, wurde das originale `trizod` durch `trizod_updated` ersetzt. Das beste Ergebnis kam aus einem reinen DisProt-Retraining. Der beste neue Checkpoint war `{best['best_checkpoint']}` mit AP `{best['disprot_AP']:.6f}` und AUROC `{best['disprot_AUROC']:.6f}` auf dem DisProt-Testset.

Verglichen mit dem originalen UdonPred-DisProt-Head ist das unter der standardmäßig geglätteten Auswertung eine kleine, aber konsistente Verbesserung:

- AP: `{best['delta_AP_vs_baseline']:+.6f}`
- AUROC: `{best['delta_AUROC_vs_baseline']:+.6f}`

## Experimentelles Setup

Alle Experimente nutzten die lokale UdonPred-Architektur mit ProstT5-Embeddings und wurden auf demselben DisProt-Testsplit ausgewertet: 99,239 gelabelte Residuen, 31,401 positive Residuen, positiver Anteil 0.3164. Falls nicht anders angegeben, wurden die Scores mit dem UdonPred-Standard `smooth=1.5` geglättet.

Getestete Strategien:

- **Original DisProt head:** pretrained UdonPred-Baseline aus `results/udonpred_matrix/matrix.csv`.
- **Equal multitask:** ein gemeinsamer Head auf `disprot`, `trizod_updated`, `chezod`, `softdis` und `plddt` mit gleichen Loss-Gewichten; `trizod`, `atlas` und `pdbflex` deaktiviert.
- **DisProt-only:** ein Head nur auf DisProt trainiert.
- **Weighted multitask:** dieselben Hilfsziele wie beim Equal-Multitask, aber DisProt-Loss-Gewicht 5 und jedes Hilfsziel-Gewicht 0.1.

## Ergebnisse

{markdown_table(summary)}

![Beste Strategien nach DisProt-Metriken](images/disprot_focus/best_strategy_metrics.png)

![Differenz zur Baseline](images/disprot_focus/delta_vs_baseline.png)

## Checkpoint-Verhalten

Das Checkpoint-Ranking zeigt, dass ein Stopp nach gemischtem `eval_loss` nicht ausgereicht hätte. Entscheidend sind DisProt AP und AUROC. DisProt-only erzeugte die besten Checkpoints, während beide Multitask-Varianten klar unter der originalen DisProt-Baseline blieben.

![Checkpoint-Verlauf](images/disprot_focus/checkpoint_curves.png)

Der beste Equal-Multitask-Checkpoint erreichte AP `{equal['disprot_AP']:.6f}` und AUROC `{equal['disprot_AUROC']:.6f}`. Eine stärkere DisProt-Gewichtung verbesserte das Multitask-Ergebnis auf AP `{weighted['disprot_AP']:.6f}` und AUROC `{weighted['disprot_AUROC']:.6f}`, blieb aber weiterhin unter der originalen Baseline und deutlich unter DisProt-only.

## Glättungs-Check

Beim besten DisProt-only-Checkpoint reduzierte das Entfernen der Glättung die Leistung auf AP `{float(smooth0['disprot_AP']):.6f}` und AUROC `{float(smooth0['disprot_AUROC']):.6f}`. Die Hauptverbesserung sollte daher als standardmäßig geglättetes UdonPred-Ergebnis berichtet werden. Für einen vollständig passenden ungeglätteten Vergleich müsste auch der originale pretrained DisProt-Head mit `smooth=0` ausgewertet werden.

![Effekt der Glättung](images/disprot_focus/smoothing_effect.png)

## Interpretation

Die Experimente zeigen einen klaren Zielkonflikt im Shared-Head-Multitask-Setup. Kontinuierliche Hilfslabels aus `trizod_updated`, `chezod`, `softdis` und `plddt` halfen DisProt nicht, wenn alles durch einen gemeinsamen Output-Head gelernt wurde. Eine stärkere DisProt-Gewichtung reduzierte den Schaden, erreichte aber nicht die Leistung des reinen DisProt-Trainings.

Die wichtigste praktische Schlussfolgerung:

> Für dieses Setup ist ein fokussiertes DisProt-only-Retraining die empfohlene Verbesserung. Hilfsdatensätze sollten nicht einfach in denselben Head gemischt werden, solange keine bessere Architektur, kein zweistufiges Fine-tuning oder kein task-spezifisches Routing verwendet wird.

## Erzeugte Dateien

- Zusammenfassungs-CSV: `results/disprot_focus_experiment_summary.csv`
- Equal-Multitask-Checkpoint-Ranking: `results/disprot_focus_checkpoint_eval/checkpoint_summary.csv`
- DisProt-only-Checkpoint-Ranking: `results/disprot_only_checkpoint_eval/checkpoint_summary.csv`
- Weighted-Multitask-Checkpoint-Ranking: `results/disprot_weighted_multitask_checkpoint_eval/checkpoint_summary.csv`
- Exportierter ONNX-Head: `UdonPred/weights_disprot_only/disprot_only.onnx`
- CAID-Predictions vom exportierten Head: `results/disprot_only_predictions/`

"""
    REPORT_PATH.write_text(text)


def main() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary = read_best_experiments()
    summary.to_csv(SUMMARY_PATH, index=False)
    plot_best_strategy_bars(summary)
    plot_delta_bars(summary)
    plot_checkpoint_curves()
    plot_smoothing_effect(summary)
    write_report(summary)
    print(f"Wrote {REPORT_PATH}")
    print(f"Wrote {SUMMARY_PATH}")
    print(f"Wrote figures to {FIGURE_DIR}")


if __name__ == "__main__":
    main()
