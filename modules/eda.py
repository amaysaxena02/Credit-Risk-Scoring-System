"""
Module 3: Exploratory Data Analysis (EDA)
Generates and saves key visualizations to outputs/eda/.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

PALETTE = {'0': '#2ECC71', '1': '#E74C3C'}
PLOT_STYLE = 'dark_background'


def run_eda(df: pd.DataFrame, output_dir: str = 'outputs/eda') -> None:
    """Generate all EDA plots and save them to output_dir."""
    os.makedirs(output_dir, exist_ok=True)
    plt.style.use(PLOT_STYLE)

    _plot_target_distribution(df, output_dir)
    _plot_loan_amount_by_status(df, output_dir)
    _plot_annual_income_distribution(df, output_dir)
    _plot_default_rate_by_grade(df, output_dir)
    _plot_correlation_heatmap(df, output_dir)
    _plot_purpose_breakdown(df, output_dir)

    logger.info(f"EDA plots saved to {output_dir}/")


def _plot_target_distribution(df, out):
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df['target'].value_counts().sort_index()
    labels = ['Non-Default (0)', 'Default (1)']
    colors = ['#2ECC71', '#E74C3C']
    bars = ax.bar(labels, counts.values, color=colors, edgecolor='white', linewidth=0.8)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + counts.max() * 0.01,
                f'{val:,}', ha='center', va='bottom', color='white', fontsize=11)
    ax.set_title('Target Class Distribution', fontsize=14, pad=12)
    ax.set_ylabel('Count')
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, out, '1_target_distribution.png')


def _plot_loan_amount_by_status(df, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    loan = pd.to_numeric(df['loan_amnt'], errors='coerce')
    for target_val, label, color in [(0, 'Non-Default', '#2ECC71'), (1, 'Default', '#E74C3C')]:
        data = loan[df['target'] == target_val].dropna()
        ax.hist(data, bins=50, alpha=0.6, label=label, color=color, edgecolor='none')
    ax.set_title('Loan Amount Distribution by Default Status', fontsize=14)
    ax.set_xlabel('Loan Amount ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, out, '2_loan_amount_dist.png')


def _plot_annual_income_distribution(df, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    inc = pd.to_numeric(df['annual_inc'], errors='coerce')
    cap = inc.quantile(0.99)
    inc = inc.clip(upper=cap)
    for target_val, label, color in [(0, 'Non-Default', '#2ECC71'), (1, 'Default', '#E74C3C')]:
        data = inc[df['target'] == target_val].dropna()
        ax.hist(data, bins=50, alpha=0.6, label=label, color=color, edgecolor='none')
    ax.set_title('Annual Income Distribution (99th pct cap)', fontsize=14)
    ax.set_xlabel('Annual Income ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    ax.spines[['top', 'right']].set_visible(False)
    _save(fig, out, '3_annual_income_dist.png')


def _plot_default_rate_by_grade(df, out):
    if 'grade' not in df.columns:
        return
    grade_stats = (
        df.groupby('grade')['target']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('grade')
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(grade_stats['grade'], grade_stats['mean'] * 100,
                  color=plt.cm.RdYlGn_r(np.linspace(0.2, 0.9, len(grade_stats))),
                  edgecolor='white', linewidth=0.6)
    ax.set_title('Default Rate by Loan Grade', fontsize=14)
    ax.set_xlabel('Grade')
    ax.set_ylabel('Default Rate (%)')
    ax.spines[['top', 'right']].set_visible(False)
    for bar, row in zip(bars, grade_stats.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                f'{row.mean*100:.1f}%', ha='center', va='bottom', fontsize=9, color='white')
    _save(fig, out, '4_default_rate_by_grade.png')


def _plot_correlation_heatmap(df, out):
    # Keep top 18 numeric features by |correlation with target|
    num_df = df.select_dtypes(include=[np.number])
    if 'target' in num_df.columns:
        top_cols = (
            num_df.corr()['target']
            .abs()
            .drop('target', errors='ignore')
            .nlargest(18)
            .index.tolist()
        )
        corr = num_df[top_cols].corr()
    else:
        corr = num_df.iloc[:, :18].corr()

    fig, ax = plt.subplots(figsize=(14, 11))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
        center=0, ax=ax, annot_kws={'size': 7},
        linewidths=0.3, linecolor='#1a1a2e'
    )
    ax.set_title('Feature Correlation Heatmap (Top 18 by |corr with target|)', fontsize=13)
    plt.tight_layout()
    _save(fig, out, '5_correlation_heatmap.png')


def _plot_purpose_breakdown(df, out):
    if 'purpose' not in df.columns:
        return
    purpose_rate = (
        df.groupby('purpose')['target']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values('mean', ascending=False)
    )
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, len(purpose_rate)))
    ax.barh(purpose_rate['purpose'], purpose_rate['mean'] * 100, color=colors)
    ax.set_title('Default Rate by Loan Purpose', fontsize=14)
    ax.set_xlabel('Default Rate (%)')
    ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    plt.tight_layout()
    _save(fig, out, '6_purpose_breakdown.png')


def _save(fig, out_dir, filename):
    path = os.path.join(out_dir, filename)
    fig.savefig(path, dpi=120, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Saved: {path}")
