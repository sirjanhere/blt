# Copyright (c) Meta Platforms, Inc. and affiliates.
import json
import os
import sys
from pathlib import Path

import altair as alt
import pandas as pd
from omegaconf import OmegaConf
from pydantic import BaseModel


class PlotEntropiesConfig(BaseModel):
    data_path: str | None
    chart_path: str
    score_override_path: str | None = None
    threshold_override: float | None = None

    class Config:
        extra = "forbid"


class PlotEntropiesData(BaseModel):
    text: str
    threshold: float = 1.335442066192627
    dataframe_json: str | None

    class Config:
        extra = "forbid"

# --- ADDED: Helper function to compute metrics ---
def compute_metrics(df):
    # True positives (TP): predicted AND ground truth
    tp = ((df['predicted_start'] == 1) & (df['ground_truth_start'] == 1)).sum()
    # False positives (FP): predicted but not ground truth
    fp = ((df['predicted_start'] == 1) & (df['ground_truth_start'] == 0)).sum()
    # False negatives (FN): not predicted but ground truth
    fn = ((df['predicted_start'] == 0) & (df['ground_truth_start'] == 1)).sum()
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    return precision, recall, f1, tp, fp, fn

def main():
    config_path = sys.argv[1]
    file_config = OmegaConf.load(config_path)
    # Omit program name and config file name
    cli_conf = OmegaConf.from_cli(sys.argv[2:])
    conf_dict = OmegaConf.to_container(
        OmegaConf.merge(file_config, cli_conf), resolve=True, throw_on_missing=True
    )
    plot_config = PlotEntropiesConfig(**conf_dict)
    with open(plot_config.data_path) as f:
        json_data = f.read()

    plot_data = PlotEntropiesData.model_validate_json(json_data)
    df = pd.read_json(plot_data.dataframe_json)
    print("LEN", len(df))
    if plot_config.threshold_override is None:
        threshold = plot_data.threshold
    else:
        threshold = plot_config.threshold_override
    if plot_config.score_override_path is not None:
        with open(plot_config.score_override_path) as f:
            scores = json.load(f)["score"]
            assert len(scores) == len(df)
            df["entropies"] = scores
            # Use entropy thresholding for predicted boundaries
            df["predicted_start"] = [1] + (df["entropies"] > threshold).values.tolist()[:-1]
    else:
        # If not overridden, try to use a "predicted_start" column if it exists, else create it from threshold
        if "predicted_start" not in df.columns:
            df["predicted_start"] = [1] + (df["entropies"] > threshold).values.tolist()[:-1]

    # --- ADDED: Ensure 'ground_truth_start' exists ---
    if "ground_truth_start" not in df.columns:
        raise ValueError("Dataframe must contain 'ground_truth_start' column marking sentence starts.")

    # --- ADDED: Compute metrics ---
    precision, recall, f1, tp, fp, fn = compute_metrics(df)
    print(f"Precision: {precision:.4f}  Recall: {recall:.4f}  F1: {f1:.4f}  TP: {tp}  FP: {fp}  FN: {fn}")

    x_ticks = []
    for row in df.itertuples():
        position = row.position
        token = row.tokens
        x_ticks.append(f"{str(position).zfill(3)}|{token}")
    df["position_with_token"] = x_ticks
    print(df)

    x_axis = alt.Axis(
        labelExpr="split(datum.label, '|')[1]",
        grid=False,
        labelOverlap=False,
        labelAngle=0,
    )
    width = 1200
    height = 220  # increased for annotation room

    base = alt.Chart(df).properties(width=width, height=height)

    # Entropy line with points
    points = base.mark_line(point=True).encode(
        x=alt.X("position_with_token:O", title=None, axis=x_axis),
        y=alt.Y(
            "entropies",
            title="Entropy of Next Byte",
        ),
    )

    # Entropy threshold line
    rule = base.mark_rule(color="red", strokeDash=[4, 4]).encode(
        y=alt.datum(threshold),
    )

    # --- CHANGED: Patch rules (predicted boundary starts) are now blue dashed ---
    pred_rules = (
        alt.Chart(df[df["predicted_start"] > 0])
        .mark_rule(color="blue", strokeDash=[4, 2])
        .encode(x=alt.X("position_with_token:O", axis=x_axis))
    )
    # --- ADDED: Ground truth sentence starts, green solid ---
    gt_rules = (
        alt.Chart(df[df["ground_truth_start"] > 0])
        .mark_rule(color="green", strokeDash=[1, 0])
        .encode(x=alt.X("position_with_token:O", axis=x_axis))
    )

    # --- ADDED: Text annotation for F1, precision, recall ---
    text = alt.Chart(pd.DataFrame({
        'text': [f'F1: {f1:.3f}   Precision: {precision:.3f}   Recall: {recall:.3f}'],
        'x': [0], 'y': [df['entropies'].max() + 0.3]  # place above the highest entropy
    })).mark_text(
        align='left', baseline='top', fontSize=18, color="black"
    ).encode(
        x=alt.value(10),  # pixels from left
        y=alt.value(10),  # pixels from top
        text='text'
    )

    # Combine all chart elements
    chart = gt_rules + pred_rules + rule + points + text
    chart = chart.configure_axis(labelFontSize=15, titleFontSize=15)
    path = Path(plot_config.chart_path)
    path.parent.mkdir(exist_ok=True)
    chart.save(path)


if __name__ == "__main__":
    main()