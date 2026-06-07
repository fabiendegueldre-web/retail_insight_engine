import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(
    page_title="Retail Insight Engine",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Retail Insight Engine")
st.caption("Upload sales data and generate charts, forecasts, benchmarks and commercial recommendations.")

uploaded_file = st.file_uploader(
    "Upload your retail sales file",
    type=["csv", "xlsx", "xls"]
)


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
        raise ValueError("Could not read the uploaded file. Please check the file format.")

    return best_df


def detect_column(df, keywords):
    for col in df.columns:
        clean_col = str(col).lower().replace("_", " ").replace("-", " ")
        if any(k in clean_col for k in keywords):
            return col
    return None


def clean_currency(series):
    return (
        series.astype(str)
        .str.replace("£", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.replace("€", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("%", "", regex=False)
        .replace("nan", np.nan)
        .replace("", np.nan)
        .astype(float)
    )


def generate_insights(df, date_col, sales_col, product_col, category_col, store_col, units_col, margin_col):
    insights = []

    total_sales = df[sales_col].sum()
    insights.append(f"Total sales were {total_sales:,.0f}.")

    if date_col:
        monthly = df.groupby(pd.Grouper(key=date_col, freq="M"))[sales_col].sum().dropna()
        if len(monthly) >= 2 and monthly.iloc[-2] != 0:
            change = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
            direction = "increased" if change >= 0 else "declined"
            insights.append(f"Sales {direction} by {abs(change):.1f}% versus the previous month.")

    if category_col:
        top_cat = df.groupby(category_col)[sales_col].sum().sort_values(ascending=False).head(1)
        if not top_cat.empty:
            insights.append(f"The strongest category was {top_cat.index[0]}, generating {top_cat.iloc[0]:,.0f} in sales.")

    if product_col:
        top_product = df.groupby(product_col)[sales_col].sum().sort_values(ascending=False).head(1)
        if not top_product.empty:
            insights.append(f"The top product was {top_product.index[0]}, generating {top_product.iloc[0]:,.0f} in sales.")

    if store_col:
        store_perf = df.groupby(store_col)[sales_col].sum().sort_values()
        if not store_perf.empty:
            insights.append(f"The lowest-performing store was {store_perf.index[0]}, with {store_perf.iloc[0]:,.0f} in sales.")
            insights.append(f"The best-performing store was {store_perf.index[-1]}, with {store_perf.iloc[-1]:,.0f} in sales.")

    if margin_col:
        avg_margin = df[margin_col].mean()
        insights.append(f"Average margin was {avg_margin:.1f}%.")

    return insights


def create_forecast(df, date_col, sales_col):
    daily = df.groupby(date_col)[sales_col].sum().reset_index()
    daily = daily.sort_values(date_col)
    daily = daily.dropna()

    daily["day_number"] = np.arange(len(daily))

    X = daily[["day_number"]]
    y = daily[sales_col]

    model = LinearRegression()
    model.fit(X, y)

    future_days = 30

    future = pd.DataFrame({
        "day_number": np.arange(len(daily), len(daily) + future_days)
    })

    last_date = daily[date_col].max()

    future[date_col] = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=future_days
    )

    future[sales_col] = model.predict(future[["day_number"]])
    future[sales_col] = future[sales_col].clip(lower=0)

    return daily, future


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

    date_col = detect_column(df, ["date", "week", "month", "day"])
    sales_col = detect_column(df, ["sales", "revenue", "turnover", "amount", "value", "net"])
    product_col = detect_column(df, ["product", "sku", "item", "model"])
    category_col = detect_column(df, ["category", "department", "range", "segment"])
    store_col = detect_column(df, ["store", "location", "branch", "retailer", "customer"])
    units_col = detect_column(df, ["units", "quantity", "qty", "volume"])
    margin_col = detect_column(df, ["margin", "gm", "profit"])

    st.subheader("2. Column mapping")

    cols = list(df.columns)
    empty_options = [None] + cols

    date_col = st.selectbox(
        "Date column",
        empty_options,
        index=empty_options.index(date_col) if date_col in empty_options else 0
    )

    sales_col = st.selectbox(
        "Sales / revenue column",
        cols,
        index=cols.index(sales_col) if sales_col in cols else 0
    )

    product_col = st.selectbox(
        "Product column",
        empty_options,
        index=empty_options.index(product_col) if product_col in empty_options else 0
    )

    category_col = st.selectbox(
        "Category column",
        empty_options,
        index=empty_options.index(category_col) if category_col in empty_options else 0
    )

    store_col = st.selectbox(
        "Store / location column",
        empty_options,
        index=empty_options.index(store_col) if store_col in empty_options else 0
    )

    units_col = st.selectbox(
        "Units column",
        empty_options,
        index=empty_options.index(units_col) if units_col in empty_options else 0
    )

    margin_col = st.selectbox(
        "Margin % column",
        empty_options,
        index=empty_options.index(margin_col) if margin_col in empty_options else 0
    )

    try:
        df[sales_col] = clean_currency(df[sales_col])
    except Exception:
        st.error("The selected sales column could not be converted into numbers. Please select another sales/revenue column.")
        st.stop()

    df = df.dropna(subset=[sales_col])

    if units_col:
        df[units_col] = pd.to_numeric(df[units_col], errors="coerce")

    if margin_col:
        df[margin_col] = clean_currency(df[margin_col])

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.dropna(subset=[date_col])

    st.subheader("3. Executive dashboard")

    total_sales = df[sales_col].sum()
    avg_sales = df[sales_col].mean()
    total_units = df[units_col].sum() if units_col else None
    avg_margin = df[margin_col].mean() if margin_col else None

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    kpi1.metric("Total sales", f"{total_sales:,.0f}")
    kpi2.metric("Average sale", f"{avg_sales:,.0f}")
    kpi3.metric("Units sold", f"{total_units:,.0f}" if units_col else "N/A")
    kpi4.metric("Average margin", f"{avg_margin:.1f}%" if margin_col else "N/A")

    if date_col:
        sales_trend = df.groupby(date_col)[sales_col].sum().reset_index()

        fig = px.line(
            sales_trend,
            x=date_col,
            y=sales_col,
            title="Sales trend over time"
        )
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        if category_col:
            category_sales = (
                df.groupby(category_col)[sales_col]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )

            fig = px.bar(
                category_sales,
                x=sales_col,
                y=category_col,
                orientation="h",
                title="Sales by category"
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        if store_col:
            store_sales = (
                df.groupby(store_col)[sales_col]
                .sum()
                .sort_values(ascending=False)
                .head(20)
                .reset_index()
            )

            fig = px.bar(
                store_sales,
                x=sales_col,
                y=store_col,
                orientation="h",
                title="Sales by store / customer"
            )
            st.plotly_chart(fig, use_container_width=True)

    if product_col:
        st.subheader("4. Product performance")

        product_sales = (
            df.groupby(product_col)[sales_col]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        top_products = product_sales.head(10)
        bottom_products = product_sales.tail(10)

        p1, p2 = st.columns(2)

        with p1:
            fig = px.bar(
                top_products,
                x=sales_col,
                y=product_col,
                orientation="h",
                title="Top 10 products"
            )
            st.plotly_chart(fig, use_container_width=True)

        with p2:
            fig = px.bar(
                bottom_products,
                x=sales_col,
                y=product_col,
                orientation="h",
                title="Bottom 10 products"
            )
            st.plotly_chart(fig, use_container_width=True)

    if store_col and category_col:
        st.subheader("5. Store and category benchmark")

        benchmark = (
            df.groupby([store_col, category_col])[sales_col]
            .sum()
            .reset_index()
        )

        fig = px.density_heatmap(
            benchmark,
            x=category_col,
            y=store_col,
            z=sales_col,
            title="Store/category sales heatmap"
        )
        st.plotly_chart(fig, use_container_width=True)

    if date_col and len(df[date_col].dropna().unique()) >= 5:
        st.subheader("6. 30-day sales forecast")

        actual, forecast = create_forecast(df, date_col, sales_col)

        forecast_chart = pd.concat([
            actual[[date_col, sales_col]].assign(Type="Actual"),
            forecast[[date_col, sales_col]].assign(Type="Forecast")
        ])

        fig = px.line(
            forecast_chart,
            x=date_col,
            y=sales_col,
            color="Type",
            title="Sales forecast"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Forecasting needs a valid date column with at least 5 different dates.")

    st.subheader("7. Automated conclusions")

    insights = generate_insights(
        df,
        date_col,
        sales_col,
        product_col,
        category_col,
        store_col,
        units_col,
        margin_col
    )

    for insight in insights:
        st.info(insight)

    st.subheader("8. Commercial recommendations")

    recommendations = []

    if category_col:
        recommendations.append(
            "Focus commercial activity on the strongest category and identify whether its success can be replicated across weaker categories."
        )

    if store_col:
        recommendations.append(
            "Review the lowest-performing store or customer for execution gaps, stock availability, training quality and promotional compliance."
        )

    if product_col:
        recommendations.append(
            "Use the top 10 products as hero lines in future promotional planning and attach-rate campaigns."
        )

    if margin_col:
        recommendations.append(
            "Compare high-sales products against margin contribution to avoid over-prioritising low-profit volume."
        )

    if date_col:
        recommendations.append(
            "Use the sales forecast to plan stock, staffing and promotional intensity for the next 30 days."
        )

    for rec in recommendations:
        st.success(rec)

    st.subheader("9. Download analysed data")

    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download cleaned sales data",
        data=csv,
        file_name="cleaned_retail_sales_data.csv",
        mime="text/csv"
    )

else:
    st.warning("Upload a CSV or Excel file to begin.")