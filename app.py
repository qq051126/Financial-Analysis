import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import streamlit as st

try:
    import wrds
except Exception:
    wrds = None

st.set_page_config(page_title="WRDS Financial Analysis Dashboard", layout="wide")

COLORS = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0"]


# =========================
# Data / WRDS helper funcs
# =========================
def connect_wrds(username: str | None = None):
    if wrds is None:
        raise ImportError(
            "The 'wrds' package is not installed. Install it first with: pip install wrds"
        )
    try:
        if username:
            return wrds.Connection(wrds_username=username)
        return wrds.Connection()
    except Exception as e:
        raise RuntimeError(
            "Failed to connect to WRDS. Please make sure your WRDS account and local setup are available."
        ) from e


def get_table_columns(db, schema, table):
    sql = f"""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = '{schema}'
          AND table_name   = '{table}'
        ORDER BY ordinal_position
    """
    return db.raw_sql(sql)["column_name"].tolist()


def pick_col(columns, candidates):
    for c in candidates:
        if c in columns:
            return c
    return None


def clean_tickers(raw_input):
    tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]
    if len(tickers) != 3:
        raise ValueError("Please enter exactly 3 tickers separated by commas, e.g. NKE,LULU,UAA")
    return tickers


def validate_dates(start, end):
    s = pd.to_datetime(start, format="%Y-%m-%d", errors="raise")
    e = pd.to_datetime(end, format="%Y-%m-%d", errors="raise")
    if s > e:
        raise ValueError("Start date must be earlier than or equal to end date.")
    return start, end


def fetch_company_raw(db, tickers, start_date, end_date):
    funda_cols = get_table_columns(db, "comp", "funda")

    tic_col = pick_col(funda_cols, ["tic", "ticker"])
    date_col = pick_col(funda_cols, ["datadate"])
    gvkey_col = pick_col(funda_cols, ["gvkey"])
    at_col = pick_col(funda_cols, ["at"])
    sale_col = pick_col(funda_cols, ["sale", "revt"])
    ni_col = pick_col(funda_cols, ["ni", "ib"])
    ceq_col = pick_col(funda_cols, ["ceq"])

    for col, name in [
        (tic_col, "tic"),
        (date_col, "datadate"),
        (gvkey_col, "gvkey"),
        (at_col, "at"),
        (sale_col, "sale"),
        (ni_col, "ni"),
        (ceq_col, "ceq"),
    ]:
        if col is None:
            raise ValueError(f"Required field '{name}' not found in comp.funda.")

    ticker_list = "', '".join(tickers)
    sql = f"""
        SELECT
            {gvkey_col}  AS gvkey,
            {date_col}   AS datadate,
            {tic_col}    AS tic,
            {at_col}     AS at,
            {sale_col}   AS sale,
            {ni_col}     AS ni,
            {ceq_col}    AS ceq
        FROM comp.funda
        WHERE {tic_col}  IN ('{ticker_list}')
          AND {date_col} BETWEEN '{start_date}' AND '{end_date}'
          AND indfmt = 'INDL'
          AND datafmt = 'STD'
          AND popsrc = 'D'
          AND consol = 'C'
        ORDER BY {tic_col}, {date_col}
    """
    df = db.raw_sql(sql)
    if df.empty:
        raise ValueError("No company data returned. Check your tickers and date range.")
    return df


def fetch_industry_code(db, gvkeys):
    company_cols = get_table_columns(db, "comp", "company")
    gvkey_col = pick_col(company_cols, ["gvkey"])
    ind_field = pick_col(company_cols, ["sic", "sich", "gind", "ggroup", "gsector"])

    if gvkey_col is None or ind_field is None:
        raise ValueError("comp.company does not expose a usable gvkey/industry field.")

    gvkey_list = "', '".join(gvkeys)
    sql = f"""
        SELECT {gvkey_col} AS gvkey, {ind_field} AS ind_code
        FROM comp.company
        WHERE {gvkey_col} IN ('{gvkey_list}')
    """
    df = db.raw_sql(sql).dropna(subset=["ind_code"])
    if df.empty:
        raise ValueError("No industry code found for the selected companies.")

    modal_code = df["ind_code"].mode().iloc[0]
    return ind_field, modal_code


def fetch_industry_raw(db, ind_field, ind_code, start_date, end_date):
    funda_cols = get_table_columns(db, "comp", "funda")
    company_cols = get_table_columns(db, "comp", "company")

    f_gvkey = pick_col(funda_cols, ["gvkey"])
    f_date = pick_col(funda_cols, ["datadate"])
    f_at = pick_col(funda_cols, ["at"])
    f_sale = pick_col(funda_cols, ["sale", "revt"])
    f_ni = pick_col(funda_cols, ["ni", "ib"])
    f_ceq = pick_col(funda_cols, ["ceq"])
    c_gvkey = pick_col(company_cols, ["gvkey"])

    if ind_field not in company_cols:
        raise ValueError(f"Industry field '{ind_field}' missing from comp.company.")

    code_expr = f"'{ind_code}'" if isinstance(ind_code, str) else str(int(ind_code))

    sql = f"""
        SELECT
            f.{f_gvkey}  AS gvkey,
            f.{f_date}   AS datadate,
            f.{f_at}     AS at,
            f.{f_sale}   AS sale,
            f.{f_ni}     AS ni,
            f.{f_ceq}    AS ceq
        FROM comp.funda f
        INNER JOIN comp.company c
            ON f.{f_gvkey} = c.{c_gvkey}
        WHERE c.{ind_field} = {code_expr}
          AND f.{f_date} BETWEEN '{start_date}' AND '{end_date}'
          AND f.indfmt  = 'INDL'
          AND f.datafmt = 'STD'
          AND f.popsrc  = 'D'
          AND f.consol  = 'C'
        ORDER BY f.{f_gvkey}, f.{f_date}
    """
    df = db.raw_sql(sql)
    if df.empty:
        raise ValueError("No industry peer data returned for the detected industry code.")
    return df


def clean_raw(df):
    df = df.copy()
    df["datadate"] = pd.to_datetime(df["datadate"])
    df["year"] = df["datadate"].dt.year

    if "tic" in df.columns:
        df["tic"] = df["tic"].astype(str).str.upper().str.strip()
    if "gvkey" in df.columns:
        df["gvkey"] = df["gvkey"].astype(str).str.strip()

    df.dropna(subset=["at", "sale", "ni", "ceq"], inplace=True)
    df = df[(df["at"] > 0) & (df["ceq"] != 0) & (df["sale"] > 0)]

    if "tic" in df.columns:
        df = df.sort_values("datadate").drop_duplicates(subset=["tic", "year"], keep="last")
    else:
        df = df.sort_values("datadate").drop_duplicates(subset=["gvkey", "year"], keep="last")

    return df.reset_index(drop=True)


def compute_ratios(df, group_col="tic"):
    df = df.sort_values([group_col, "year"]).copy()
    df["roe"] = df["ni"] / df["ceq"] * 100
    df["roa"] = df["ni"] / df["at"] * 100
    df["profit_margin"] = df["ni"] / df["sale"] * 100
    df["asset_turnover"] = df["sale"] / df["at"]
    df["rev_growth"] = df.groupby(group_col)["sale"].pct_change() * 100

    ratio_cols = ["roe", "roa", "profit_margin", "asset_turnover", "rev_growth"]
    for col in ratio_cols:
        lo, hi = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(lo, hi)

    return df


def aggregate_company(df):
    ratio_cols = [
        "roe", "roa", "profit_margin", "asset_turnover", "rev_growth", "at", "sale", "ni"
    ]
    return df.groupby("tic")[ratio_cols].mean().reset_index()


def aggregate_industry(df):
    ratio_cols = [
        "roe", "roa", "profit_margin", "asset_turnover", "rev_growth", "at", "sale", "ni"
    ]
    yearly = df.groupby("year")[ratio_cols].median().reset_index()
    summary_row = yearly[ratio_cols].mean().to_frame().T
    summary_row["tic"] = "Industry Avg"
    return yearly, summary_row


def build_summary_table(company_summary, industry_summary_row):
    cols = [
        "tic", "roe", "roa", "profit_margin", "asset_turnover", "rev_growth", "at", "sale", "ni"
    ]
    return pd.concat([company_summary[cols], industry_summary_row[cols]], ignore_index=True)


@st.cache_data(show_spinner=False)
def run_pipeline(wrds_username, tickers, industry_name, start_date, end_date):
    db = connect_wrds(wrds_username)
    try:
        company_raw = fetch_company_raw(db, tickers, start_date, end_date)
        gvkeys = company_raw["gvkey"].dropna().astype(str).unique().tolist()
        ind_field, ind_code = fetch_industry_code(db, gvkeys)
        industry_raw = fetch_industry_raw(db, ind_field, ind_code, start_date, end_date)

        company_clean = clean_raw(company_raw)
        industry_clean = clean_raw(industry_raw)

        company_ratios = compute_ratios(company_clean, group_col="tic")
        industry_ratios = compute_ratios(industry_clean, group_col="gvkey")

        company_summary = aggregate_company(company_ratios)
        industry_yearly, ind_row = aggregate_industry(industry_ratios)
        summary_df = build_summary_table(company_summary, ind_row)

        meta = {
            "industry_name": industry_name,
            "industry_field": ind_field,
            "industry_code": ind_code,
            "company_rows": len(company_raw),
            "industry_rows": len(industry_raw),
        }
        return company_raw, industry_raw, company_clean, industry_clean, company_ratios, industry_yearly, summary_df, meta
    finally:
        try:
            db.close()
        except Exception:
            pass


# =========================
# Plot helpers
# =========================
def _fmt_millions(ax, axis="y"):
    fmt = mticker.FuncFormatter(lambda x, _: f"${x:,.0f}M")
    if axis == "y":
        ax.yaxis.set_major_formatter(fmt)
    else:
        ax.xaxis.set_major_formatter(fmt)


def plot_line_chart(company_df, industry_yearly, industry_name):
    fig, ax = plt.subplots(figsize=(11, 6))
    tickers = company_df["tic"].unique()
    for i, tic in enumerate(tickers):
        sub = company_df[company_df["tic"] == tic].sort_values("year")
        ax.plot(sub["year"].astype(int), sub["sale"], marker="o", color=COLORS[i], label=tic, linewidth=2)

    ind = industry_yearly.sort_values("year")
    ax.plot(
        ind["year"].astype(int),
        ind["sale"],
        marker="s",
        color=COLORS[3],
        linestyle="--",
        linewidth=2,
        label=f"{industry_name} Industry Median",
    )

    all_years = sorted(company_df["year"].astype(int).unique())
    ax.set_xticks(all_years)
    ax.set_xticklabels(all_years, rotation=45, ha="right")
    ax.set_title("Revenue (Net Sales) Over Time", fontsize=14, fontweight="bold")
    ax.set_xlabel("Fiscal Year")
    ax.set_ylabel("Net Sales (USD millions)")
    _fmt_millions(ax, "y")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_scatter_chart(company_df, industry_yearly, industry_name):
    fig, ax = plt.subplots(figsize=(9, 7))
    tickers = company_df["tic"].unique()
    for i, tic in enumerate(tickers):
        sub = company_df[company_df["tic"] == tic]
        size = np.clip(sub["sale"] / sub["sale"].max() * 300, 30, 300)
        ax.scatter(sub["roa"], sub["roe"], s=size, color=COLORS[i], alpha=0.75,
                   label=tic, edgecolors="white", linewidth=0.5)
        for _, row in sub.iterrows():
            ax.annotate(str(int(row["year"])), (row["roa"], row["roe"]), textcoords="offset points", xytext=(5, 3), fontsize=7)

    ind = industry_yearly
    ax.scatter(ind["roa"], ind["roe"], s=120, color=COLORS[3], marker="D",
               label=f"{industry_name} Industry Median", zorder=5, edgecolors="black", linewidth=0.8)
    ax.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.axvline(0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("ROA vs. ROE by Year (bubble size ∝ revenue)", fontsize=14, fontweight="bold")
    ax.set_xlabel("Return on Assets — ROA (%)")
    ax.set_ylabel("Return on Equity — ROE (%)")
    ax.legend()
    ax.grid(linestyle="--", alpha=0.35)
    plt.tight_layout()
    return fig


def plot_bar_chart(summary_df):
    entities = summary_df["tic"].tolist()
    metrics = ["roe", "roa", "profit_margin"]
    labels = ["ROE (%)", "ROA (%)", "Profit Margin (%)"]
    x = np.arange(len(entities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for j, (metric, label) in enumerate(zip(metrics, labels)):
        vals = summary_df[metric].values
        bars = ax.bar(x + (j - 1) * width, vals, width=width * 0.9, label=label)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, f"{v:.1f}%",
                    ha="center", va="bottom", fontsize=7.5)

    ax.set_title("Average Profitability Ratios: Companies vs Industry", fontsize=14, fontweight="bold")
    ax.set_xlabel("Entity")
    ax.set_ylabel("Ratio (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(entities)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()
    return fig


def plot_box_chart(company_df):
    ratio_info = [
        ("roe", "ROE (%)"),
        ("roa", "ROA (%)"),
        ("profit_margin", "Profit Margin (%)"),
        ("asset_turnover", "Asset Turnover (times)"),
    ]
    tickers = company_df["tic"].unique()
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()

    for ax, (col, ylabel) in zip(axes, ratio_info):
        data = [company_df.loc[company_df["tic"] == t, col].dropna().values for t in tickers]
        bp = ax.boxplot(data, patch_artist=True, notch=False, medianprops={"color": "black", "linewidth": 2})
        for patch, color in zip(bp["boxes"], COLORS):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_xticklabels(tickers)
        ax.set_title(ylabel, fontsize=12, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

    fig.suptitle("Distribution of Financial Ratios by Company (annual observations)", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    return fig


def plot_radar_chart(summary_df):
    metrics = ["roe", "roa", "profit_margin", "asset_turnover", "rev_growth"]
    metric_labels = ["ROE (%)", "ROA (%)", "Profit Margin (%)", "Asset Turnover (×)", "Revenue Growth (%)"]
    plot_df = summary_df.set_index("tic")[metrics].copy()

    for col in metrics:
        lo, hi = plot_df[col].min(), plot_df[col].max()
        plot_df[col] = 0.5 if lo == hi else (plot_df[col] - lo) / (hi - lo)

    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    for i, entity in enumerate(plot_df.index):
        vals = plot_df.loc[entity, metrics].tolist() + [plot_df.loc[entity, metrics[0]]]
        color = COLORS[i] if i < len(COLORS) else "#607D8B"
        ax.plot(angles, vals, color=color, linewidth=2, label=entity)
        ax.fill(angles, vals, color=color, alpha=0.10)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")
    ax.set_title("Radar Chart — Five Financial Dimensions (normalised)", fontsize=13, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15), fontsize=10)
    plt.tight_layout()
    return fig


# =========================
# UI
# =========================
st.title("WRDS Financial Analysis Dashboard")
st.caption("Built from your notebook logic: company data → industry peers → cleaning → ratios → 5 charts")

with st.sidebar:
    st.header("Input")
    wrds_username = st.text_input("WRDS username (optional)", value="")
    raw_tickers = st.text_input("3 U.S. tickers", value="NKE,LULU,UAA")
    industry_name = st.text_input("Industry display name", value="Sportswear & Apparel")
    start_date = st.text_input("Start date", value="2015-01-01")
    end_date = st.text_input("End date", value="2023-12-31")
    run_btn = st.button("Run analysis", type="primary")

st.markdown(
    """
This app compares **3 U.S. companies** with their **industry peers** using WRDS Compustat annual data.
It calculates ROE, ROA, profit margin, asset turnover, and revenue growth, then shows five visualizations.
"""
)

with st.expander("What this app needs"):
    st.write(
        "- Access to WRDS\n"
        "- Python packages: streamlit, wrds, pandas, numpy, matplotlib\n"
        "- Internet / institutional access if your WRDS setup requires it"
    )

if run_btn:
    try:
        tickers = clean_tickers(raw_tickers)
        start_date, end_date = validate_dates(start_date, end_date)

        with st.spinner("Running WRDS query and building charts..."):
            (
                company_raw,
                industry_raw,
                company_clean,
                industry_clean,
                company_ratios,
                industry_yearly,
                summary_df,
                meta,
            ) = run_pipeline(wrds_username, tickers, industry_name, start_date, end_date)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Tickers", ", ".join(tickers))
        c2.metric("Industry field", str(meta["industry_field"]))
        c3.metric("Industry code", str(meta["industry_code"]))
        c4.metric("Company rows", int(meta["company_rows"]))

        st.subheader("Summary table")
        display_df = summary_df[["tic", "roe", "roa", "profit_margin", "asset_turnover", "rev_growth"]].rename(
            columns={
                "tic": "Entity",
                "roe": "ROE (%)",
                "roa": "ROA (%)",
                "profit_margin": "Profit Margin (%)",
                "asset_turnover": "Asset Turnover (×)",
                "rev_growth": "Revenue Growth (%)",
            }
        )
        st.dataframe(display_df.round(2), use_container_width=True)
        st.download_button(
            "Download summary CSV",
            data=display_df.to_csv(index=False).encode("utf-8"),
            file_name="financial_summary.csv",
            mime="text/csv",
        )

        st.subheader("Charts")
        st.pyplot(plot_line_chart(company_ratios, industry_yearly, industry_name), use_container_width=True)
        st.pyplot(plot_scatter_chart(company_ratios, industry_yearly, industry_name), use_container_width=True)
        st.pyplot(plot_bar_chart(summary_df), use_container_width=True)
        st.pyplot(plot_box_chart(company_ratios), use_container_width=True)
        st.pyplot(plot_radar_chart(summary_df), use_container_width=True)

        with st.expander("Preview cleaned company data"):
            st.dataframe(company_clean, use_container_width=True)
        with st.expander("Preview cleaned industry peer data"):
            st.dataframe(industry_clean.head(200), use_container_width=True)

    except Exception as e:
        st.error(str(e))
        st.info(
            "If the app cannot connect, the most common reasons are: WRDS package missing, WRDS login not configured, or no access to WRDS from this environment."
        )
else:
    st.info("Enter your inputs in the sidebar, then click 'Run analysis'.")
