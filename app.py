from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots


BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"


@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    defense = pd.read_csv(DATA_DIR / "defense-industry.csv").rename(
        columns={"MarketVector Global Defense Industry": "defense_index"}
    )
    defense["Date"] = pd.to_datetime(defense["Date"], format="%m/%Y")

    gold = pd.read_csv(DATA_DIR / "gold.csv").rename(columns={"Price": "gold_price"})
    gold["Date"] = pd.to_datetime(gold["Date"], format="%Y-%m")

    defense = defense.sort_values("Date").reset_index(drop=True)
    gold = gold.sort_values("Date").reset_index(drop=True)

    shared = defense.merge(gold, on="Date", how="inner")
    shared["defense_norm"] = shared["defense_index"] / shared["defense_index"].iloc[0] * 100
    shared["gold_norm"] = shared["gold_price"] / shared["gold_price"].iloc[0] * 100
    shared["defense_return"] = shared["defense_index"].pct_change()
    shared["gold_return"] = shared["gold_price"].pct_change()

    return defense, gold, shared


def format_month(timestamp: pd.Timestamp) -> str:
    return timestamp.strftime("%b %Y")


def build_normalized_chart(frame: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["defense_norm"],
            name="Defense industry",
            mode="lines",
            line={"color": "#1e3a8a", "width": 3},
        )
    )
    figure.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["gold_norm"],
            name="Gold",
            mode="lines",
            line={"color": "#b8860b", "width": 3},
        )
    )
    figure.update_layout(
        title="Indexed performance over time",
        xaxis_title="Month",
        yaxis_title="Indexed to 100 at the selected start month",
        hovermode="x unified",
        legend_title_text="Series",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def build_absolute_chart(frame: pd.DataFrame) -> go.Figure:
    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["defense_index"],
            name="Defense industry index",
            mode="lines",
            line={"color": "#1e3a8a", "width": 3},
        ),
        secondary_y=False,
    )
    figure.add_trace(
        go.Scatter(
            x=frame["Date"],
            y=frame["gold_price"],
            name="Gold price (USD/oz)",
            mode="lines",
            line={"color": "#b8860b", "width": 3},
        ),
        secondary_y=True,
    )
    figure.update_layout(
        title="Absolute levels",
        hovermode="x unified",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    figure.update_xaxes(title_text="Month")
    figure.update_yaxes(title_text="Defense industry index", secondary_y=False)
    figure.update_yaxes(title_text="Gold price (USD/oz)", secondary_y=True)
    return figure


def build_returns_scatter(frame: pd.DataFrame) -> go.Figure:
    scatter = frame.dropna(subset=["defense_return", "gold_return"]).copy()

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=scatter["defense_return"],
            y=scatter["gold_return"],
            text=scatter["Date"].dt.strftime("%b %Y"),
            mode="markers",
            name="Monthly observations",
            marker={
                "size": 10,
                "color": "#1e3a8a",
                "opacity": 0.75,
                "line": {"color": "#f8fafc", "width": 1},
            },
            hovertemplate=(
                "<b>%{text}</b><br>"
                "Defense return: %{x:.2%}<br>"
                "Gold return: %{y:.2%}<extra></extra>"
            ),
        )
    )

    if len(scatter) >= 2:
        slope, intercept = np.polyfit(scatter["defense_return"], scatter["gold_return"], 1)
        x_range = np.linspace(scatter["defense_return"].min(), scatter["defense_return"].max(), 100)
        y_range = slope * x_range + intercept
        figure.add_trace(
            go.Scatter(
                x=x_range,
                y=y_range,
                mode="lines",
                name="Trend line",
                line={"color": "#b8860b", "dash": "dash", "width": 2},
                hoverinfo="skip",
            )
        )

    figure.update_layout(
        title="Monthly return relationship",
        xaxis_title="Defense industry monthly return",
        yaxis_title="Gold monthly return",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def build_rolling_correlation_chart(frame: pd.DataFrame, window: int) -> go.Figure:
    corr_frame = frame.copy()
    corr_frame["rolling_correlation"] = corr_frame["defense_return"].rolling(window).corr(
        corr_frame["gold_return"]
    )

    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=corr_frame["Date"],
            y=corr_frame["rolling_correlation"],
            mode="lines",
            name=f"{window}-month rolling correlation",
            line={"color": "#0f766e", "width": 3},
        )
    )
    figure.add_hline(y=0, line_dash="dot", line_color="#6b7280")
    figure.update_layout(
        title="Rolling correlation of monthly returns",
        xaxis_title="Month",
        yaxis_title="Correlation",
        margin={"l": 20, "r": 20, "t": 60, "b": 20},
    )
    return figure


def main() -> None:
    st.set_page_config(page_title="War and Gold", page_icon=":bar_chart:", layout="wide")

    defense, gold, shared = load_data()

    st.title("War and Gold")
    st.caption("A Streamlit dashboard tracking the relationship between gold prices and the global defense industry.")

    overlap_start = shared["Date"].min()
    overlap_end = shared["Date"].max()
    defense_end = defense["Date"].max()
    gold_end = gold["Date"].max()

    with st.sidebar:
        st.header("Controls")
        selected_range = st.select_slider(
            "Shared date range",
            options=shared["Date"].tolist(),
            value=(overlap_start, overlap_end),
            format_func=format_month,
        )
        rolling_window = st.selectbox("Rolling correlation window", options=[3, 6, 12], index=1)

    filtered = shared.loc[
        (shared["Date"] >= selected_range[0]) & (shared["Date"] <= selected_range[1])
    ].copy()

    if len(filtered) < 2:
        st.error("Select at least two months to render the dashboard.")
        return

    filtered["defense_norm"] = filtered["defense_index"] / filtered["defense_index"].iloc[0] * 100
    filtered["gold_norm"] = filtered["gold_price"] / filtered["gold_price"].iloc[0] * 100
    filtered["defense_return"] = filtered["defense_index"].pct_change()
    filtered["gold_return"] = filtered["gold_price"].pct_change()

    full_period_corr = filtered["defense_return"].corr(filtered["gold_return"])
    defense_change = filtered["defense_norm"].iloc[-1] - 100
    gold_change = filtered["gold_norm"].iloc[-1] - 100

    if gold_end > defense_end:
        st.info(
            "The gold dataset runs through "
            f"{format_month(gold_end)}, while the defense dataset runs through {format_month(defense_end)}. "
            "Combined charts use the overlapping period only."
        )

    metric_1, metric_2, metric_3, metric_4 = st.columns(4)
    metric_1.metric("Overlap period", f"{format_month(filtered['Date'].min())} to {format_month(filtered['Date'].max())}")
    metric_2.metric("Defense indexed change", f"{defense_change:+.1f}%")
    metric_3.metric("Gold indexed change", f"{gold_change:+.1f}%")
    metric_4.metric(
        "Return correlation",
        "n/a" if pd.isna(full_period_corr) else f"{full_period_corr:.2f}",
    )

    left, right = st.columns((1.6, 1))

    with left:
        st.plotly_chart(build_normalized_chart(filtered), use_container_width=True)
        st.plotly_chart(build_absolute_chart(filtered), use_container_width=True)

    with right:
        st.plotly_chart(build_rolling_correlation_chart(filtered, rolling_window), use_container_width=True)
        st.plotly_chart(build_returns_scatter(filtered), use_container_width=True)

    st.subheader("Underlying monthly data")
    st.dataframe(
        filtered.assign(
            Month=filtered["Date"].dt.strftime("%Y-%m"),
            **{
                "Defense industry index": filtered["defense_index"].round(2),
                "Gold price (USD/oz)": filtered["gold_price"].round(2),
                "Defense monthly return": filtered["defense_return"].map(
                    lambda value: None if pd.isna(value) else round(value * 100, 2)
                ),
                "Gold monthly return": filtered["gold_return"].map(
                    lambda value: None if pd.isna(value) else round(value * 100, 2)
                ),
            },
        )[
            [
                "Month",
                "Defense industry index",
                "Gold price (USD/oz)",
                "Defense monthly return",
                "Gold monthly return",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )


if __name__ == "__main__":
    main()
