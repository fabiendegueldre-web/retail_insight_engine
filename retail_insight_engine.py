import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Vendor Growth Opportunity Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Vendor Growth Opportunity Engine")
st.caption("Upload vendor sell-out data and generate product scores, account scorecards and commercial recommendations.")

uploaded_file = st.file_uploader(
    "Upload your vendor sell-out file",
    type=["csv", "xlsx", "xls"]
)


# -----------------------------
# File loading
# -----------------------------

def load_file(file):
    if file.name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file)

    encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252", "utf-16"]
    separators = [",", ";", "\t", "|"]

    best_df = None
    best_score = -1

    for encoding in encodings:
        for sep in separators:
            try:
                file.seek(0)
                temp_df = pd.read_csv(
                    file,
                    encoding=encoding,
                    sep=sep,
                    low_memory=False,
                    on_bad_lines="skip"
                )

                temp_df = temp_df.dropna(how="all")
                temp_df = temp_df.dropna(axis=1, how="all")

                score = temp_df.shape[0] * temp_df.shape[1]

                if score > best_score:
                    best_score = score
                    best_df = temp_df

            except Exception:
                continue

    if best_df is None or best_df.empty:
        raise ValueError("Could not read uploaded file.")

    return best_df


def detect_column(df, possible_names):
    for col in df.columns:
        clean_col = str(col).lower().strip()
        for name in possible_names:
            if name.lower() in clean_col:
                return col
    return None


def clean_number(series):
    return (
        series.astype(str)
        .str.replace("$", "", regex=False)
        .str.replace("£", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .replace("nan", np.nan)
        .replace("", np.nan)
        .astype(float)
    )


def safe_growth(current, previous):
    if previous == 0 or pd.isna(previous):
        return np.nan
    return ((current - previous) / previous) * 100


def normalise_score(series):
    if series.max() == series.min():
        return pd.Series([50] * len(series), index=series.index)
    return ((series - series.min()) / (series.max() - series.min())) * 100


def product_status(score, growth):
    if score >= 85 and growth >= 0:
        return "Strategic Hero"
    if score >= 70 and growth >= 0:
        return "Growth Driver"
    if score >= 55:
        return "Opportunity"
    if growth < -15:
        return "At Risk"
    return "Declining / Low Priority"


def action_priority(score):
    if score >= 75:
        return "🔥 Immediate Action"
    if score >= 50:
        return "🟡 Monitor"
    return "🔴 Low Priority"


# -----------------------------
# Main app
# -----------------------------

if uploaded_file:

    try:
        df = load_file(uploaded_file)
        df.columns = df.columns.astype(str).str.strip()
        df = df.dropna(how="all")
        df = df.dropna(axis=1, how="all")
    except Exception as e:
        st.error("The file could not be loaded properly.")
        st.exception(e)
        st.stop()

    st.subheader("1. Data preview")
    st.write(f"Rows: {df.shape[0]:,} | Columns: {df.shape[1]:,}")
    st.dataframe(df.head(20), use_container_width=True)

    # Auto-detect your typical fields
    country_col = detect_column(df, ["Country Name", "Country"])
    account_col = detect_column(df, ["Account Name", "Account", "Retailer", "Customer"])
    product_group_col = detect_column(df, ["Product Group Desc", "Product Group"])
    product_type_col = detect_column(df, ["Product Type Desc", "Product Type"])
    product_line_col = detect_column(df, ["Product Line Desc", "Product Line"])
    sku_col = detect_column(df, ["Product Sku", "SKU"])
    product_desc_col = detect_column(df, ["Marketing Product Description", "Product Description"])
    fiscal_year_col = detect_column(df, ["Fiscal Year", "Year"])
    fiscal_quarter_col = detect_column(df, ["Fiscal Quarter Alias", "Fiscal Quarter"])
    fiscal_month_col = detect_column(df, ["Fiscal Month Alias", "Fiscal Month"])
    sales_col = detect_column(df, ["Indirect Sales Out USD", "Sales Out", "Revenue", "Sales"])

    st.subheader("2. Vendor field mapping")

    cols = list(df.columns)

    country_col = st.selectbox("Country column", [None] + cols, index=([None] + cols).index(country_col) if country_col in cols else 0)
    account_col = st.selectbox("Account / retailer column", [None] + cols, index=([None] + cols).index(account_col) if account_col in cols else 0)
    product_group_col = st.selectbox("Product group column", [None] + cols, index=([None] + cols).index(product_group_col) if product_group_col in cols else 0)
    product_type_col = st.selectbox("Product type column", [None] + cols, index=([None] + cols).index(product_type_col) if product_type_col in cols else 0)
    product_line_col = st.selectbox("Product line column", [None] + cols, index=([None] + cols).index(product_line_col) if product_line_col in cols else 0)
    sku_col = st.selectbox("Product SKU column", [None] + cols, index=([None] + cols).index(sku_col) if sku_col in cols else 0)
    product_desc_col = st.selectbox("Marketing product description column", [None] + cols, index=([None] + cols).index(product_desc_col) if product_desc_col in cols else 0)
    fiscal_year_col = st.selectbox("Fiscal year column", [None] + cols, index=([None] + cols).index(fiscal_year_col) if fiscal_year_col in cols else 0)
    fiscal_quarter_col = st.selectbox("Fiscal quarter column", [None] + cols, index=([None] + cols).index(fiscal_quarter_col) if fiscal_quarter_col in cols else 0)
    fiscal_month_col = st.selectbox("Fiscal month column", [None] + cols, index=([None] + cols).index(fiscal_month_col) if fiscal_month_col in cols else 0)
    sales_col = st.selectbox("Indirect sales out USD column", cols, index=cols.index(sales_col) if sales_col in cols else 0)

    if not sku_col or not account_col or not fiscal_year_col or not fiscal_month_col:
        st.warning("For best results, map Account, SKU, Fiscal Year and Fiscal Month.")
        st.stop()

    df[sales_col] = clean_number(df[sales_col])
    df = df.dropna(subset=[sales_col])

    df["_Period"] = (
        df[fiscal_year_col].astype(str)
        + " - "
        + df[fiscal_month_col].astype(str)
    )

    product_name_col = product_desc_col if product_desc_col else sku_col
    category_col = product_line_col if product_line_col else product_type_col if product_type_col else product_group_col

    # Create ordered period index
    period_table = (
        df[[fiscal_year_col, fiscal_month_col, "_Period"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    period_table["_YearNum"] = pd.to_numeric(period_table[fiscal_year_col], errors="coerce")
    period_table["_MonthNum"] = pd.to_numeric(period_table[fiscal_month_col], errors="coerce")
    period_table = period_table.sort_values(["_YearNum", "_MonthNum"])

    ordered_periods = period_table["_Period"].tolist()

    if len(ordered_periods) < 2:
        st.warning("The file needs at least two fiscal months to calculate momentum and recommendations.")
        st.stop()

    latest_period = ordered_periods[-1]
    previous_period = ordered_periods[-2]

    latest_df = df[df["_Period"] == latest_period]
    previous_df = df[df["_Period"] == previous_period]

    # -----------------------------
    # Product scoring engine
    # -----------------------------

    product_latest = (
        latest_df.groupby([sku_col, product_name_col], dropna=False)[sales_col]
        .sum()
        .reset_index()
        .rename(columns={sales_col: "Latest Sales"})
    )

    product_previous = (
        previous_df.groupby([sku_col], dropna=False)[sales_col]
        .sum()
        .reset_index()
        .rename(columns={sales_col: "Previous Sales"})
    )

    product_accounts = (
        latest_df.groupby(sku_col)[account_col]
        .nunique()
        .reset_index()
        .rename(columns={account_col: "Account Coverage"})
    )

    product_scores = product_latest.merge(product_previous, on=sku_col, how="left")
    product_scores = product_scores.merge(product_accounts, on=sku_col, how="left")
    product_scores["Previous Sales"] = product_scores["Previous Sales"].fillna(0)
    product_scores["Growth %"] = product_scores.apply(
        lambda x: safe_growth(x["Latest Sales"], x["Previous Sales"]),
        axis=1
    ).fillna(0)

    product_scores["Revenue Score"] = normalise_score(product_scores["Latest Sales"])
    product_scores["Growth Score"] = normalise_score(product_scores["Growth %"])
    product_scores["Coverage Score"] = normalise_score(product_scores["Account Coverage"])

    product_scores["Product Score"] = (
        product_scores["Revenue Score"] * 0.4
        + product_scores["Growth Score"] * 0.4
        + product_scores["Coverage Score"] * 0.2
    )

    product_scores["Status"] = product_scores.apply(
        lambda x: product_status(x["Product Score"], x["Growth %"]),
        axis=1
    )

    product_scores = product_scores.sort_values("Product Score", ascending=False)

    # -----------------------------
    # Account scorecards
    # -----------------------------

    account_latest = (
        latest_df.groupby(account_col)[sales_col]
        .sum()
        .reset_index()
        .rename(columns={sales_col: "Latest Sales"})
    )

    account_previous = (
        previous_df.groupby(account_col)[sales_col]
        .sum()
        .reset_index()
        .rename(columns={sales_col: "Previous Sales"})
    )

    account_skus = (
        latest_df.groupby(account_col)[sku_col]
        .nunique()
        .reset_index()
        .rename(columns={sku_col: "SKU Count"})
    )

    account_scorecards = account_latest.merge(account_previous, on=account_col, how="left")
    account_scorecards = account_scorecards.merge(account_skus, on=account_col, how="left")
    account_scorecards["Previous Sales"] = account_scorecards["Previous Sales"].fillna(0)
    account_scorecards["Growth %"] = account_scorecards.apply(
        lambda x: safe_growth(x["Latest Sales"], x["Previous Sales"]),
        axis=1
    ).fillna(0)

    account_scorecards["Revenue Score"] = normalise_score(account_scorecards["Latest Sales"])
    account_scorecards["Growth Score"] = normalise_score(account_scorecards["Growth %"])
    account_scorecards["Breadth Score"] = normalise_score(account_scorecards["SKU Count"])

    account_scorecards["Account Health Score"] = (
        account_scorecards["Revenue Score"] * 0.4
        + account_scorecards["Growth Score"] * 0.4
        + account_scorecards["Breadth Score"] * 0.2
    )

    account_scorecards = account_scorecards.sort_values("Account Health Score", ascending=False)

    # -----------------------------
    # Opportunity engine
    # -----------------------------

    account_product_latest = (
        latest_df.groupby([account_col, sku_col, product_name_col], dropna=False)[sales_col]
        .sum()
        .reset_index()
        .rename(columns={sales_col: "Account Product Sales"})
    )

    product_average = (
        account_product_latest.groupby(sku_col)["Account Product Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Account Product Sales": "Average Account Sales"})
    )

    opportunities = account_product_latest.merge(product_average, on=sku_col, how="left")
    opportunities = opportunities.merge(
        product_scores[[sku_col, "Product Score", "Growth %", "Status"]],
        on=sku_col,
        how="left"
    )

    opportunities["Gap To Average"] = opportunities["Average Account Sales"] - opportunities["Account Product Sales"]
    opportunities["Gap To Average"] = opportunities["Gap To Average"].clip(lower=0)

    opportunities["Opportunity Score"] = (
        normalise_score(opportunities["Gap To Average"]) * 0.45
        + opportunities["Product Score"].fillna(0) * 0.35
        + normalise_score(opportunities["Growth %"].fillna(0)) * 0.20
    )

    opportunities["Priority"] = opportunities["Opportunity Score"].apply(action_priority)

    opportunities["Recommended Action"] = opportunities.apply(
        lambda x: (
            f"Increase promotional support for {x[product_name_col]} in {x[account_col]}. "
            f"This product under-indexes versus the account average and has a product score of {x['Product Score']:.0f}."
        ),
        axis=1
    )

    opportunities = opportunities.sort_values("Opportunity Score", ascending=False)

    # -----------------------------
    # Tabs
    # -----------------------------

    tab1, tab2, tab3, tab4 = st.tabs([
        "Commercial Action Centre",
        "Product Scoring Engine",
        "Opportunity Engine",
        "Account Scorecards"
    ])

    with tab1:
        st.subheader("🔥 Commercial Action Centre")
        st.caption(f"Latest period: {latest_period} | Previous period: {previous_period}")

        total_sales = latest_df[sales_col].sum()
        previous_sales = previous_df[sales_col].sum()
        total_growth = safe_growth(total_sales, previous_sales)

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Latest sales", f"${total_sales:,.0f}")
        k2.metric("Previous sales", f"${previous_sales:,.0f}")
        k3.metric("Growth", f"{total_growth:.1f}%" if not pd.isna(total_growth) else "N/A")
        k4.metric("Active SKUs", f"{latest_df[sku_col].nunique():,}")

        st.markdown("### Top 10 recommended commercial actions")

        action_cols = [
            "Priority",
            account_col,
            sku_col,
            product_name_col,
            "Account Product Sales",
            "Average Account Sales",
            "Gap To Average",
            "Opportunity Score",
            "Recommended Action"
        ]

        st.dataframe(
            opportunities[action_cols].head(10),
            use_container_width=True,
            hide_index=True
        )

        st.markdown("### Executive summary")

        top_account = account_scorecards.iloc[0][account_col]
        top_product = product_scores.iloc[0][product_name_col]
        top_opp = opportunities.iloc[0]

        st.info(
            f"""
            Latest sell-out was **${total_sales:,.0f}**, with growth of **{total_growth:.1f}%** versus the previous period.

            The strongest account is **{top_account}**.

            The highest scoring product is **{top_product}**.

            The top commercial action is to increase support for **{top_opp[product_name_col]}** in **{top_opp[account_col]}**, 
            with an estimated gap to average of **${top_opp['Gap To Average']:,.0f}**.
            """
        )

    with tab2:
        st.subheader("Product Scoring Engine")

        st.dataframe(
            product_scores[
                [
                    sku_col,
                    product_name_col,
                    "Latest Sales",
                    "Previous Sales",
                    "Growth %",
                    "Account Coverage",
                    "Product Score",
                    "Status"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            product_scores.head(20),
            x="Product Score",
            y=product_name_col,
            orientation="h",
            title="Top 20 products by product score"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Opportunity Scoring Engine")

        st.dataframe(
            opportunities[
                [
                    "Priority",
                    account_col,
                    sku_col,
                    product_name_col,
                    "Account Product Sales",
                    "Average Account Sales",
                    "Gap To Average",
                    "Product Score",
                    "Opportunity Score",
                    "Recommended Action"
                ]
            ].head(100),
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            opportunities.head(20),
            x="Opportunity Score",
            y=product_name_col,
            color=account_col,
            orientation="h",
            title="Top 20 product/account promotion opportunities"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Account Scorecards")

        st.dataframe(
            account_scorecards[
                [
                    account_col,
                    "Latest Sales",
                    "Previous Sales",
                    "Growth %",
                    "SKU Count",
                    "Account Health Score"
                ]
            ],
            use_container_width=True,
            hide_index=True
        )

        fig = px.bar(
            account_scorecards.head(20),
            x="Account Health Score",
            y=account_col,
            orientation="h",
            title="Account health score"
        )
        st.plotly_chart(fig, use_container_width=True)

        if category_col:
            st.markdown("### Account/category benchmark")

            benchmark = (
                latest_df.groupby([account_col, category_col])[sales_col]
                .sum()
                .reset_index()
            )

            fig = px.density_heatmap(
                benchmark,
                x=category_col,
                y=account_col,
                z=sales_col,
                title="Account/category sales heatmap"
            )
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Download outputs")

    output = opportunities.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download commercial action centre CSV",
        data=output,
        file_name="commercial_action_centre.csv",
        mime="text/csv"
    )

else:
    st.warning("Upload your vendor sell-out CSV or Excel file to begin.")