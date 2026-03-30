from pathlib import Path
import os
import re
from typing import Dict, List, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="Bone Tumor MIL — Dashboard",
    page_icon="🔬",
    layout="wide",
)

CLASS_ORDER = ["COD", "COF", "FD", "HG-OS", "JTOF", "LG-OS", "Other", "PSOF"]

# ── Log directory ──────────────────────────────────────────────────────────────
# Priority: 1) sidebar input  2) MIL_LOG_DIR env var  3) same folder as script
# Streamlit Cloud mounts repo at /mount/src/<reponame-lowercase>
BASE_DIR = Path("/mount/src/dashboard")
if not BASE_DIR.exists():
    BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = BASE_DIR / "dataset_stratified_updated.csv"

RUN_PROFILES: Dict[str, Dict[str, str]] = {
    "Random 1": {
        "subtitle": "Single-slide baseline",
        "family": "8-class comparable",
        "story": (
            "This is the clean baseline. Each patient is forced to one slide only, so the model sees "
            "the least amount of multi-slide context. It is useful because it shows what happens when "
            "the pipeline stays simple, but it also pays a price in overall performance."
        ),
        "takeaway": "Best as a baseline. Surprisingly good for FD, but weak on COF and overall score.",
    },
    "Random 2": {
        "subtitle": "2-slide averaging",
        "family": "8-class comparable",
        "story": (
            "This run is the first real jump beyond the baseline. Instead of keeping just one slide, it "
            "averages two slides for multi-slide patients. That extra context gives a clear improvement "
            "in macro-F1 and accuracy."
        ),
        "takeaway": "Strong all-round upgrade over Random 1. Best of the early random runs on overall score.",
    },
    "Random 3": {
        "subtitle": "3-slide averaging",
        "family": "8-class comparable",
        "story": (
            "This run moves to up to three slides per multi-slide patient. It does not beat the final "
            "winner, but it is a balanced middle-ground model and handles some harder classes more "
            "gracefully than the earlier runs."
        ),
        "takeaway": "Balanced profile. Not the top winner, but a good middle option.",
    },
    "Random 4": {
        "subtitle": "3-slide averaging, steadier behavior",
        "family": "8-class comparable",
        "story": (
            "This run keeps the richer three-slide idea, but its real strength is stability. The fold-to-fold "
            "variation is lower, so this one behaves more predictably even if it is not the absolute top on "
            "global macro-F1."
        ),
        "takeaway": "Most stable of the 8-class runs. Very good when you want steadier CV behavior.",
    },
    "Random 5": {
        "subtitle": "Best comparable 8-class run",
        "family": "8-class comparable",
        "story": (
            "This is the strongest fair comparison run. It uses the richest multi-slide setup in this group "
            "and reaches the best overall macro-F1 and accuracy among the 350-patient, 8-class experiments. "
            "It is not the best on every class, but it is the most balanced overall."
        ),
        "takeaway": "Best final candidate for the 8-class setting.",
    },
    "Without Others": {
        "subtitle": "7-class variant",
        "family": "Separate setting",
        "story": (
            "This run has the highest raw numbers, but it is not a fair apples-to-apples comparison. It uses "
            "300 patients and a 7-class setup without the 'Other' class, so the task is easier. It should be "
            "shown separately, not mixed into the main 8-class ranking."
        ),
        "takeaway": "Highest raw score, but keep it separate because the setup is different.",
    },
}

KNOWN_SIGNATURES = {
    "Random 1": {"bags": 350, "patients": 350, "macro": 0.72609, "acc": 0.737143},
    "Random 2": {"bags": 420, "patients": 350, "macro": 0.748175, "acc": 0.751429},
    "Random 3": {"bags": 464, "patients": 350, "macro": 0.742725, "acc": 0.751429},
    "Random 4": {"bags": 491, "patients": 350, "macro": 0.743638, "acc": 0.748571},
    "Random 5": {"bags": 514, "patients": 350, "macro": 0.758702, "acc": 0.762857},
    "Without Others": {"bags": 389, "patients": 300, "macro": 0.767318, "acc": 0.783333},
}

FILENAME_HINTS = {
    "embeds_aumc-1303515": "Random 1",
    "slurm-1303334": "Random 2",
    "slurm-1313465": "Random 3",
    "slurm-1313466": "Random 4",
    "embeds_aumc-1347562": "Random 5",
    "slurm-1337865": "Without Others",
}


def _safe_round_match(a: Optional[float], b: float, tol: float = 0.0025) -> bool:
    return a is not None and abs(a - b) <= tol


def load_data() -> Optional[pd.DataFrame]:
    if not CSV_PATH.exists():
        return None
    df = pd.read_csv(CSV_PATH, encoding="latin-1")
    df.columns = df.columns.str.strip()
    df["category"] = df["category"].astype(str).str.strip()
    df["Cohort"] = df["Cohort"].astype(str).str.strip()
    df["slide"] = df["slide"].astype(str).str.strip()
    df["pt_file_count"] = pd.to_numeric(df["pt_file_count"], errors="coerce").fillna(0)
    return df


def _search(pattern: str, text: str, flags: int = re.S):
    match = re.search(pattern, text, flags)
    return match.groups() if match else None


def _extract_dataset_info(text: str):
    bags = patients = n_classes = None
    extra = ""
    line1 = re.search(
        r"Dataset built — mode:.*?\n\s*([0-9]+) bags \| ([0-9]+) unique patients(?: \((.*?)\))?",
        text, re.S,
    )
    if line1:
        bags = int(line1.group(1))
        patients = int(line1.group(2))
        extra = line1.group(3) or ""
    line2 = re.search(r"\n\s*([0-9]+) classes:\s*\[(.*?)\]", text)
    if line2:
        n_classes = int(line2.group(1))
    elif "without 'Other'" in text:
        n_classes = 7
    return bags, patients, extra, n_classes


def classify_run(parsed: dict) -> Optional[str]:
    file_name = (parsed.get("file") or "").lower()
    script = parsed.get("script") or ""
    extra = (parsed.get("extra") or "").lower()
    sampling = (parsed.get("sampling") or "").lower()
    bags = parsed.get("bags")
    patients = parsed.get("global_num_patients") or parsed.get("patients")
    macro = parsed.get("global_macro_f1")
    acc = parsed.get("global_accuracy")

    for hint, label in FILENAME_HINTS.items():
        if hint.lower() in file_name:
            return label

    if script == "MIL_slide_bags.py" or "one slide randomly selected" in extra:
        return "Random 1"
    if patients == 300:
        return "Without Others"
    if "2-slide averaging" in sampling:
        return "Random 2"

    for label, sig in KNOWN_SIGNATURES.items():
        bags_ok = bags == sig.get("bags") if sig.get("bags") is not None else True
        patients_ok = patients == sig["patients"]
        metric_ok = _safe_round_match(macro, sig["macro"]) and _safe_round_match(acc, sig["acc"])
        if (bags_ok and patients_ok) or (patients_ok and metric_ok):
            return label

    return None


def parse_log(path: Path) -> Optional[dict]:
    text = path.read_text(encoding="utf-8", errors="ignore")

    if "Starting MIL training" not in text and "Multi-Head Attention MIL" not in text:
        return None

    script_match = _search(r"^\s+(.+?\.py)\s+—", text, re.M)
    script_name = script_match[0].strip() if script_match else "Unknown"

    averaging_match = _search(r"^\s+(.*slide averaging for multi-slide patients)", text, re.M)
    averaging_desc = averaging_match[0].strip() if averaging_match else "Single slide per patient"

    bags, patients, extra, n_classes = _extract_dataset_info(text)

    cv_match = _search(
        r"f1_macro\s*:\s*([0-9.]+)\s*±\s*([0-9.]+).*?"
        r"f1_weighted\s*:\s*([0-9.]+)\s*±\s*([0-9.]+).*?"
        r"acc\s*:\s*([0-9.]+)\s*±\s*([0-9.]+)",
        text,
    )
    if not cv_match:
        return None

    global_match = _search(
        r"GLOBAL CV METRICS.*?accuracy\s+macro_f1\s+weighted_f1\s+num_patients\s*\n"
        r"(?:[^\n]*\n)?"  # optional separator line (e.g. ====)
        r"\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)",
        text,
    )
    if not global_match:
        return None

    class_scores: Dict[str, float] = {}
    if "Per-class report:" in text:
        section = text.split("Per-class report:", 1)[1]
        for line in section.splitlines():
            row = re.match(r"\s*([A-Za-z0-9\-]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9]+)\s*$", line)
            if row:
                cls = row.group(1)
                if cls in CLASS_ORDER:
                    class_scores[cls] = float(row.group(4))

    parsed = {
        "file": path.name,
        "script": script_name,
        "sampling": averaging_desc,
        "bags": bags,
        "patients": patients,
        "extra": extra,
        "n_classes": n_classes,
        "cv_macro_f1": float(cv_match[0]),
        "cv_macro_f1_std": float(cv_match[1]),
        "cv_weighted_f1": float(cv_match[2]),
        "cv_weighted_f1_std": float(cv_match[3]),
        "cv_accuracy": float(cv_match[4]),
        "cv_accuracy_std": float(cv_match[5]),
        "global_accuracy": float(global_match[0]),
        "global_macro_f1": float(global_match[1]),
        "global_weighted_f1": float(global_match[2]),
        "global_num_patients": int(global_match[3]),
        "class_scores": class_scores,
    }

    label = classify_run(parsed)
    if label is None:
        return None

    profile = RUN_PROFILES[label]
    parsed.update({
        "label": label,
        "subtitle": profile["subtitle"],
        "family": profile["family"],
        "story": profile["story"],
        "takeaway": profile["takeaway"],
    })
    return parsed


def load_runs() -> List[dict]:
    candidates = []
    for path in sorted(BASE_DIR.iterdir()):
        if path.suffix.lower() not in {".log", ".out"}:
            continue
        parsed = parse_log(path)
        if parsed:
            candidates.append(parsed)

    deduped: Dict[str, dict] = {}
    preferred_order = ["Random 1", "Random 2", "Random 3", "Random 4", "Random 5", "Without Others"]
    for run in candidates:
        label = run["label"]
        current = deduped.get(label)
        if current is None:
            deduped[label] = run
        else:
            cur_pat = current.get("global_num_patients") or 0
            new_pat = run.get("global_num_patients") or 0
            if new_pat > cur_pat or run.get("global_macro_f1", 0) > current.get("global_macro_f1", 0):
                deduped[label] = run

    runs = [deduped[label] for label in preferred_order if label in deduped]
    return runs


# ─────────────────────────────────────────────
#  TAB 1: DATASET HEATMAP
# ─────────────────────────────────────────────

def render_heatmap_tab() -> None:
    st.subheader("Dataset heatmap")
    # Debug: show where we are looking
    df = load_data()
    if df is None:
        st.info(
            "dataset_stratified_updated.csv is not in the same folder as this dashboard, "
            "so the heatmap tab is disabled."
        )
        return

    col1, col2 = st.columns([1, 2])
    with col1:
        selected_metric = st.radio("Show heatmap for:", ["Patients", "Slides"])
        selected_cohorts = st.multiselect(
            "Cohort",
            options=sorted(df["Cohort"].dropna().unique()),
            default=sorted(df["Cohort"].dropna().unique()),
        )
        selected_classes = st.multiselect(
            "Tumor type",
            options=[c for c in CLASS_ORDER if c in df["category"].unique()],
            default=[c for c in CLASS_ORDER if c in df["category"].unique()],
        )

    dff = df[df["Cohort"].isin(selected_cohorts) & df["category"].isin(selected_classes)].copy()
    if dff.empty:
        st.warning("No data matches the current filters.")
        return

    with col2:
        if selected_metric == "Patients":
            pivot_data = dff.groupby(["category", "Cohort"])["slide"].count().unstack(fill_value=0)
            pivot_data = pivot_data.reindex([c for c in CLASS_ORDER if c in dff["category"].unique()])
            # Add row totals
            pivot_data["TOTAL"] = pivot_data.sum(axis=1)
            fig = px.imshow(
                pivot_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Teal",
                labels=dict(x="Cohort", y="Tumor type", color="Patients"),
                height=500,
            )
        else:
            pivot_data = dff.groupby(["category", "Cohort"])["pt_file_count"].sum().unstack(fill_value=0)
            pivot_data = pivot_data.reindex([c for c in CLASS_ORDER if c in dff["category"].unique()])
            # Add row totals
            pivot_data["TOTAL"] = pivot_data.sum(axis=1)
            fig = px.imshow(
                pivot_data,
                text_auto=True,
                aspect="auto",
                color_continuous_scale="Oranges",
                labels=dict(x="Cohort", y="Tumor type", color="Slides"),
                height=500,
            )
        fig.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    # Summary bar chart: total per tumor type
    st.markdown("#### Total per tumor type")
    if selected_metric == "Patients":
        totals = dff.groupby("category")["slide"].count().reindex(
            [c for c in CLASS_ORDER if c in dff["category"].unique()]
        )
        y_label = "Patient count"
    else:
        totals = dff.groupby("category")["pt_file_count"].sum().reindex(
            [c for c in CLASS_ORDER if c in dff["category"].unique()]
        )
        y_label = "Slide count"

    fig_tot = px.bar(
        x=totals.index.tolist(),
        y=totals.values.tolist(),
        labels={"x": "Tumor type", "y": y_label},
        color=totals.values.tolist(),
        color_continuous_scale="Teal" if selected_metric == "Patients" else "Oranges",
        text_auto=True,
        height=320,
    )
    fig_tot.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
    fig_tot.update_traces(textposition="outside")
    st.plotly_chart(fig_tot, use_container_width=True)


# ─────────────────────────────────────────────
#  TAB 2: MIL RUN COMPARISON
# ─────────────────────────────────────────────

def render_story_card(run: dict) -> None:
    st.markdown(f"### {run['label']}")
    caption_bits = [run["subtitle"], run["script"]]
    if run.get("bags") is not None:
        caption_bits.append(f"{run['bags']} bags")
    caption_bits.append(f"{run['global_num_patients']} patients")
    if run.get("n_classes") is not None:
        caption_bits.append(f"{run['n_classes']} classes")
    st.caption(" · ".join(caption_bits))

    st.write(run["story"])

    class_scores = run["class_scores"]
    comparable_class_scores = {k: v for k, v in class_scores.items() if k in CLASS_ORDER}
    top_classes = sorted(comparable_class_scores.items(), key=lambda x: x[1], reverse=True)[:3]
    weak_classes = sorted(comparable_class_scores.items(), key=lambda x: x[1])[:2]

    strengths = ", ".join([f"{cls} ({score:.3f})" for cls, score in top_classes]) if top_classes else "N/A"
    watchouts = ", ".join([f"{cls} ({score:.3f})" for cls, score in weak_classes]) if weak_classes else "N/A"

    c1, c2, c3 = st.columns(3)
    c1.metric("Macro-F1", f"{run['global_macro_f1']:.4f}")
    c2.metric("Weighted-F1", f"{run['global_weighted_f1']:.4f}")
    c3.metric("Accuracy", f"{run['global_accuracy']:.4f}")

    st.markdown(f"**Strength classes:** {strengths}")
    st.markdown(f"**Watch-out classes:** {watchouts}")
    st.markdown(f"**Read it like this:** {run['takeaway']}")
    st.caption(f"Source file: {run['file']}")


def render_run_comparison_tab() -> None:
    runs = load_runs()

    st.subheader("MIL run comparison")
    if not runs:
        st.warning("No MIL run logs with recognizable CV metrics were found next to this dashboard.")
        st.info("Put the MIL .log/.out files in the same folder as dashboard.py and reopen the app.")
        return

    comparable = [r for r in runs if r["family"] == "8-class comparable"]

    # ── Quick summary metrics ──
    st.markdown("#### Quick reading")
    st.write(
        "The point of this section is not just to rank runs, but to tell the story of what changed: "
        "single-slide baseline, then richer multi-slide averaging, then the final comparable winner."
    )

    if comparable:
        best_overall = max(comparable, key=lambda x: x["global_macro_f1"])
        most_stable = min(comparable, key=lambda x: x["cv_macro_f1_std"])
        c1, c2, c3 = st.columns(3)
        c1.metric("Best comparable run", best_overall["label"], f"Macro-F1 {best_overall['global_macro_f1']:.4f}")
        c2.metric("Most stable comparable run", most_stable["label"], f"CV std {most_stable['cv_macro_f1_std']:.4f}")
        c3.metric("Recognized runs", str(len(runs)))

    # ── Summary table ──
    summary_rows = []
    for run in runs:
        summary_rows.append({
            "Run": run["label"],
            "Subtitle": run["subtitle"],
            "Family": run["family"],
            "Script": run["script"],
            "Sampling": run["sampling"],
            "Patients": run["global_num_patients"],
            "Macro-F1": round(run["global_macro_f1"], 4),
            "Weighted-F1": round(run["global_weighted_f1"], 4),
            "Accuracy": round(run["global_accuracy"], 4),
            "CV Macro std": round(run["cv_macro_f1_std"], 4),
            "File": run["file"],
        })
    summary_df = pd.DataFrame(summary_rows)
    st.dataframe(summary_df, use_container_width=True, hide_index=True)

    # ── Grouped bar chart: 3 metrics per run ──
    score_df = summary_df[["Run", "Macro-F1", "Weighted-F1", "Accuracy", "Family"]].melt(
        id_vars=["Run", "Family"], var_name="Metric", value_name="Score"
    )
    fig_scores = px.bar(
        score_df, x="Run", y="Score", color="Metric",
        barmode="group", height=420,
        color_discrete_sequence=["#378ADD", "#1D9E75", "#EF9F27"],
    )
    fig_scores.update_layout(
        margin=dict(t=30, b=30),
        yaxis=dict(range=[0.68, 0.80], title="Score"),
    )
    st.plotly_chart(fig_scores, use_container_width=True)

    # ── Slide count progression (comparable runs only) ──
    if comparable:
        st.markdown("#### Macro-F1 vs slide budget (8-class comparable only)")
        prog_df = pd.DataFrame([
            {"Run": r["label"], "Bags": r["bags"], "Macro-F1": r["global_macro_f1"]}
            for r in comparable if r.get("bags")
        ]).sort_values("Bags")
        fig_prog = px.line(
            prog_df, x="Bags", y="Macro-F1", text="Run",
            markers=True, height=320,
            labels={"Bags": "Number of bags (proxy for slide budget)", "Macro-F1": "Global macro-F1"},
        )
        fig_prog.update_traces(textposition="top center")
        fig_prog.update_layout(margin=dict(t=10, b=10))
        st.plotly_chart(fig_prog, use_container_width=True)

    # ── Run-by-run storyline (expandable) ──
    st.markdown("#### Run-by-run storyline")
    default_expanded = {"Random 1", "Random 5", "Without Others"}
    for run in runs:
        with st.expander(f"{run['label']} — {run['subtitle']}", expanded=(run["label"] in default_expanded)):
            render_story_card(run)

    # ── Class-by-class heatmap ──
    st.markdown("#### Class-by-class F1 heatmap")
    class_rows = []
    for run in runs:
        for cls in CLASS_ORDER:
            if cls in run["class_scores"]:
                class_rows.append({"Run": run["label"], "Class": cls, "F1": run["class_scores"][cls]})
    class_df = pd.DataFrame(class_rows)

    if not class_df.empty:
        heatmap_df = class_df.pivot(index="Class", columns="Run", values="F1")
        heatmap_df = heatmap_df.reindex([c for c in CLASS_ORDER if c in heatmap_df.index])
        run_order = [r["label"] for r in runs]
        heatmap_df = heatmap_df[[c for c in run_order if c in heatmap_df.columns]]
        fig_heatmap = px.imshow(
            heatmap_df,
            text_auto=".3f",
            aspect="auto",
            color_continuous_scale="Viridis",
            labels=dict(x="Run", y="Class", color="F1"),
            height=500,
        )
        fig_heatmap.update_layout(margin=dict(t=20, b=20))
        st.plotly_chart(fig_heatmap, use_container_width=True)

        # Best run per class table
        best_class_rows = []
        for cls in [c for c in CLASS_ORDER if c in class_df["Class"].unique()]:
            cls_slice = class_df[class_df["Class"] == cls].sort_values("F1", ascending=False)
            if not cls_slice.empty:
                winner = cls_slice.iloc[0]
                best_class_rows.append({
                    "Class": cls,
                    "Best run": winner["Run"],
                    "Best F1": round(float(winner["F1"]), 4),
                })
        st.dataframe(pd.DataFrame(best_class_rows), use_container_width=True, hide_index=True)

        # ── Per-tumor-type mean F1 across all 8-class runs ──
        st.markdown("#### Mean F1 per tumor type (8-class runs only)")
        comp_class_df = class_df[class_df["Run"].isin([r["label"] for r in comparable])]
        if not comp_class_df.empty:
            mean_per_class = (
                comp_class_df.groupby("Class")["F1"]
                .mean()
                .reindex([c for c in CLASS_ORDER if c in comp_class_df["Class"].unique()])
                .reset_index()
            )
            mean_per_class.columns = ["Class", "Mean F1"]
            fig_mean = px.bar(
                mean_per_class, x="Class", y="Mean F1",
                color="Mean F1", color_continuous_scale="Viridis",
                text_auto=".3f", height=340,
                labels={"Mean F1": "Mean F1 (across 5 comparable runs)"},
            )
            fig_mean.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
            fig_mean.update_traces(textposition="outside")
            st.plotly_chart(fig_mean, use_container_width=True)

    # ── Final conclusion ──
    st.markdown("#### Plain-language conclusion")
    st.write(
        "If you want one fair 8-class model to present as the main result, use **Random 5**. "
        "If you want a clean baseline, use **Random 1**. If you want to show the benefit of adding more "
        "slide context step by step, the nicest story is **Random 1 → Random 2 → Random 3/4 → Random 5**. "
        "Keep **Without Others** separate because it is a different 7-class, 300-patient setting."
    )


# ─────────────────────────────────────────────
#  TAB 3: PER-RUN DEEP DIVE
# ─────────────────────────────────────────────

def render_per_run_tab() -> None:
    runs = load_runs()
    if not runs:
        st.warning("No runs found.")
        return

    run_labels = [r["label"] for r in runs]
    selected_label = st.selectbox("Select run", run_labels)
    run = next(r for r in runs if r["label"] == selected_label)

    # Header
    if run["family"] == "Separate setting":
        st.warning(
            "This run uses 300 patients and 7 classes (no 'Other'). "
            "Numbers are **not** directly comparable to the 8-class runs."
        )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"### {run['label']} — {run['subtitle']}")
        caption_bits = [run["script"]]
        if run.get("bags"):
            caption_bits.append(f"{run['bags']} bags")
        caption_bits.append(f"{run['global_num_patients']} patients")
        if run.get("n_classes"):
            caption_bits.append(f"{run['n_classes']} classes")
        st.caption(" · ".join(caption_bits))
        st.write(run["story"])
        st.info(f"**Key takeaway:** {run['takeaway']}")
    with col2:
        st.metric("Macro-F1", f"{run['global_macro_f1']:.4f}")
        st.metric("Weighted-F1", f"{run['global_weighted_f1']:.4f}")
        st.metric("Accuracy", f"{run['global_accuracy']:.4f}")
        st.metric("CV Macro std", f"{run['cv_macro_f1_std']:.4f}", help="Lower = more stable across folds")

    # Per-class F1 bar chart
    if run["class_scores"]:
        st.markdown("#### Per-class F1")
        cls_df = pd.DataFrame(
            [(cls, run["class_scores"][cls]) for cls in CLASS_ORDER if cls in run["class_scores"]],
            columns=["Class", "F1"],
        ).sort_values("F1", ascending=True)

        fig_cls = px.bar(
            cls_df, x="F1", y="Class", orientation="h",
            color="F1", color_continuous_scale="Viridis",
            text_auto=".3f", height=380,
        )
        fig_cls.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10))
        fig_cls.update_traces(textposition="outside")
        st.plotly_chart(fig_cls, use_container_width=True)

        # Comparison with other comparable runs
        all_runs = load_runs()
        comparable_runs = [r for r in all_runs if r["family"] == "8-class comparable"]
        if len(comparable_runs) > 1:
            st.markdown("#### This run vs 8-class comparable average")
            mean_f1 = {}
            for cls in CLASS_ORDER:
                scores = [r["class_scores"].get(cls) for r in comparable_runs if cls in r["class_scores"]]
                if scores:
                    mean_f1[cls] = sum(scores) / len(scores)

            comparison_rows = []
            for cls in CLASS_ORDER:
                this_f1 = run["class_scores"].get(cls)
                avg_f1 = mean_f1.get(cls)
                if this_f1 is not None and avg_f1 is not None:
                    comparison_rows.append({
                        "Class": cls,
                        f"{run['label']} F1": round(this_f1, 4),
                        "Mean (comparable)": round(avg_f1, 4),
                        "Δ vs mean": round(this_f1 - avg_f1, 4),
                    })
            comp_df = pd.DataFrame(comparison_rows)

            # Radar / grouped bar
            fig_comp = go.Figure()
            fig_comp.add_trace(go.Bar(
                name=run["label"],
                x=comp_df["Class"], y=comp_df[f"{run['label']} F1"],
                marker_color="#378ADD",
            ))
            fig_comp.add_trace(go.Bar(
                name="Mean comparable",
                x=comp_df["Class"], y=comp_df["Mean (comparable)"],
                marker_color="#B4B2A9",
            ))
            fig_comp.update_layout(barmode="group", height=340, margin=dict(t=10, b=10))
            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.caption(f"Source file: {run['file']}")


# ─────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────



# ─────────────────────────────────────────────
#  TAB 4: EXTERNAL VALIDATION (GENERALIZABILITY)
# ─────────────────────────────────────────────

# Cross-site transfer results (hardcoded — no log files for these)
EXTERNAL_RESULTS = {
    "UMCG → BTC+Basel": {
        "description": "Weights trained on UMCG, evaluated on BTC + Basel dataset",
        "accuracy": 0.3357,
        "macro_f1": 0.2639,
        "weighted_f1": 0.3330,
        "n_patients": 143,
        "class_scores": {
            "COD":   {"precision": 0.375, "recall": 0.136, "f1": 0.200, "support": 22},
            "COF":   {"precision": 0.333, "recall": 0.421, "f1": 0.372, "support": 19},
            "FD":    {"precision": 0.400, "recall": 0.385, "f1": 0.392, "support": 26},
            "HG-OS": {"precision": 0.583, "recall": 0.583, "f1": 0.583, "support": 36},
            "JTOF":  {"precision": 0.000, "recall": 0.000, "f1": 0.000, "support": 12},
            "LG-OS": {"precision": 0.179, "recall": 0.227, "f1": 0.200, "support": 22},
            "PSOF":  {"precision": 0.071, "recall": 0.167, "f1": 0.100, "support": 6},
        },
    },
    "BTC+Basel → UMCG": {
        "description": "Weights trained on BTC + Basel, evaluated on UMCG dataset",
        "accuracy": 0.3086,
        "macro_f1": 0.2308,
        "weighted_f1": 0.3210,
        "n_patients": 81,
        "class_scores": {
            "COD":   {"precision": 0.000, "recall": 0.000, "f1": 0.000, "support": 1},
            "COF":   {"precision": 0.538, "recall": 0.467, "f1": 0.500, "support": 15},
            "FD":    {"precision": 0.273, "recall": 0.333, "f1": 0.300, "support": 9},
            "HG-OS": {"precision": 0.524, "recall": 0.458, "f1": 0.489, "support": 24},
            "JTOF":  {"precision": 0.000, "recall": 0.000, "f1": 0.000, "support": 10},
            "LG-OS": {"precision": 0.250, "recall": 0.158, "f1": 0.194, "support": 19},
            "PSOF":  {"precision": 0.083, "recall": 0.333, "f1": 0.133, "support": 3},
        },
    },
}

# Internal (UMCG) best run for comparison
INTERNAL_BEST = {
    "accuracy": 0.7629,
    "macro_f1": 0.7587,
    "class_f1": {
        "COD": 0.839, "COF": 0.640, "FD": 0.654,
        "HG-OS": 0.896, "JTOF": 0.750, "LG-OS": 0.800,
        "Other": 0.709, "PSOF": 0.782,
    },
    "label": "Random 5 (UMCG internal CV)",
}


def render_external_validation_tab() -> None:
    st.subheader("External validation — generalizability across sites")

    st.info(
        "These runs test whether a model trained on one site generalises to another site. "
        "The internal CV results (Random 5) are shown alongside for reference — "
        "but note that internal CV and external transfer are **not** directly comparable tasks."
    )

    # ── Direction selector ──
    directions = list(EXTERNAL_RESULTS.keys())
    selected = st.radio("Transfer direction", directions, horizontal=True)
    ext = EXTERNAL_RESULTS[selected]

    # ── Top metrics ──
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Accuracy", f"{ext['accuracy']:.4f}",
              delta=f"{ext['accuracy'] - INTERNAL_BEST['accuracy']:.4f} vs internal",
              delta_color="normal")
    c2.metric("Macro-F1", f"{ext['macro_f1']:.4f}",
              delta=f"{ext['macro_f1'] - INTERNAL_BEST['macro_f1']:.4f} vs internal",
              delta_color="normal")
    c3.metric("Weighted-F1", f"{ext['weighted_f1']:.4f}")
    c4.metric("N patients (ext)", str(ext["n_patients"]))

    st.caption(f"**Setting:** {ext['description']}")

    # ── Per-class table ──
    st.markdown("#### Per-class precision / recall / F1")
    cls_rows = []
    for cls, v in ext["class_scores"].items():
        internal_f1 = INTERNAL_BEST["class_f1"].get(cls)
        delta = f"{v['f1'] - internal_f1:.3f}" if internal_f1 is not None else "—"
        cls_rows.append({
            "Class": cls,
            "Support (ext)": v["support"],
            "Precision": round(v["precision"], 3),
            "Recall": round(v["recall"], 3),
            "F1 (external)": round(v["f1"], 3),
            "F1 (internal best)": round(internal_f1, 3) if internal_f1 else "—",
            "Δ F1": delta,
        })
    st.dataframe(pd.DataFrame(cls_rows), use_container_width=True, hide_index=True)

    # ── Per-class F1 comparison bar chart ──
    st.markdown("#### Per-class F1: external vs internal best (Random 5)")
    ext_classes = list(ext["class_scores"].keys())
    ext_f1 = [ext["class_scores"][c]["f1"] for c in ext_classes]
    int_f1 = [INTERNAL_BEST["class_f1"].get(c, 0) for c in ext_classes]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="External transfer", x=ext_classes, y=ext_f1,
                         marker_color="#D85A30", text=[f"{v:.3f}" for v in ext_f1],
                         textposition="outside"))
    fig.add_trace(go.Bar(name="Internal CV (Random 5)", x=ext_classes, y=int_f1,
                         marker_color="#B4B2A9", text=[f"{v:.3f}" for v in int_f1],
                         textposition="outside"))
    fig.update_layout(
        barmode="group", height=400,
        yaxis=dict(range=[0, 1.05], title="F1-score"),
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig, use_container_width=True)

    # ── Precision vs Recall scatter ──
    st.markdown("#### Precision vs Recall per class (external)")
    scatter_df = pd.DataFrame([
        {"Class": cls, "Precision": v["precision"], "Recall": v["recall"],
         "F1": v["f1"], "Support": v["support"]}
        for cls, v in ext["class_scores"].items()
    ])
    fig_sc = px.scatter(
        scatter_df, x="Precision", y="Recall", text="Class",
        size="Support", color="F1", color_continuous_scale="RdYlGn",
        range_x=[-0.05, 1.05], range_y=[-0.05, 1.05], height=380,
    )
    fig_sc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                     line=dict(dash="dot", color="gray", width=1))
    fig_sc.update_traces(textposition="top center")
    fig_sc.update_layout(margin=dict(t=20, b=20))
    st.plotly_chart(fig_sc, use_container_width=True)

    # ── Plain language conclusion ──
    st.markdown("#### Interpretation")
    st.warning(
        "The external macro-F1 (0.264) is substantially lower than the internal CV macro-F1 (0.759). "
        "This is expected for a single-site model applied zero-shot to a different site — staining protocols, "
        "scanner types, and patient mix all shift the feature distribution. "
        "Classes like **HG-OS** (F1 0.583) and **FD** (F1 0.392) transfer better than **JTOF** (F1 0.000) "
        "and **PSOF** (F1 0.100), which likely suffer most from domain shift or low support."
    )
    st.info(
        "**Next steps to consider:** domain adaptation, stain normalisation, or training on pooled multi-site data."
    )

st.title("🔬 Bone Tumor MIL — Dashboard")


tab1, tab2, tab3, tab4 = st.tabs(["Dataset heatmap", "MIL Run Comparison", "Per-run deep dive", "External validation"])
with tab1:
    render_heatmap_tab()
with tab2:
    render_run_comparison_tab()
with tab3:
    render_per_run_tab()
with tab4:
    render_external_validation_tab()
