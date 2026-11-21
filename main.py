import io
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import scipy.stats as stats
import streamlit as st

# ─────────────────────────────────────────────
# 한글 폰트 (Streamlit CSS)
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR&display=swap');
html, body, [class*="css"] {
   font-family: 'Noto Sans KR', 'Malgun Gothic', sans-serif;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# 페이지 설정
# ─────────────────────────────────────────────
st.set_page_config(page_title="극지식물 생육 분석", layout="wide")
st.title("🌱 우리 학교 극지식물 개체별 생육 차이 분석")

# ─────────────────────────────────────────────
# CSV 인코딩 대응 읽기 함수
# ─────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def read_csv_cached(file_bytes, encoding):
    return pd.read_csv(io.BytesIO(file_bytes), encoding=encoding)


def read_csv_uploaded(file):
    """file 업로드 객체 → 바이트를 꺼내 utf-8-sig / cp949 순서로 읽기"""
    if file is None:
        return pd.DataFrame()
    data = file.getvalue()
    for enc in ("utf-8-sig", "cp949"):
        try:
            return read_csv_cached(data, enc)
        except Exception:
            pass
    return pd.DataFrame()


# ─────────────────────────────────────────────
# 열 이름 표준화 함수 (문서 그대로 사용)
# ─────────────────────────────────────────────
def std_cols(df):
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    rename = {}
    for c in df.columns:
        k = c.lower().replace(" ", "")
        if k in ("time", "categories", "시간", "날짜", "datetime", "date"):
            rename[c] = "time"
        elif k in ("temperature", "온도"):
            rename[c] = "temperature"
        elif k in ("humidity", "습도"):
            rename[c] = "humidity"
        elif k == "ph":
            rename[c] = "ph"
        elif k == "ec":
            rename[c] = "ec"
        elif "생중량" in k:
            rename[c] = "생중량"
        elif "잎" in k:
            rename[c] = "잎 수"
        elif "지상부" in k and "길이" in k:
            rename[c] = "지상부 길이"
        elif "지하부" in k and "길이" in k:
            rename[c] = "지하부 길이"
        elif "지상부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지상부 질량"
        elif "지하부" in k and ("질량" in k or "무게" in k):
            rename[c] = "지하부 질량"
        elif "개체" in k:
            rename[c] = "개체번호"
    df = df.rename(columns=rename)
    for c in df.columns:
        if c in ("time", "개체번호"):
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            num = pd.to_numeric(
                df[c].astype(str).str.replace(",", "").str.extract(r"(-?\d+\.?\d*)")[0],
                errors="coerce",
            )
            if num.notna().mean() >= 0.5:
                df[c] = num
    if "생중량" not in df.columns and {"지상부 질량", "지하부 질량"} <= set(df.columns):
        df["생중량"] = df["지상부 질량"] + df["지하부 질량"]
    if "time" in df.columns:
        t = df["time"].astype(str).str.strip()
        parsed = pd.to_datetime(t, errors="coerce", format="mixed")
        bad = parsed.isna() & t.str.match(r"^\d{8}_\d{1,2}$")
        if bad.any():
            parts = t[bad].str.split("_", expand=True)
            parsed[bad] = (
                pd.to_datetime(parts[0], format="%Y%m%d")
                + pd.to_timedelta(parts[1].astype(int), unit="h")
            )
        df["time"] = parsed
    return df


# ─────────────────────────────────────────────
# 사이드바 ― CSV 업로드
# ─────────────────────────────────────────────
st.sidebar.header("📁 데이터 업로드")

env_file = st.sidebar.file_uploader("환경 데이터 CSV", type="csv")
growth_file = st.sidebar.file_uploader("생육 결과 데이터 CSV", type="csv")

if env_file is not None:
    env_df = read_csv_uploaded(env_file)
    env_df = std_cols(env_df)
    env_school_name = str(env_file.name).split("_")[0] if "_" in env_file.name else str(env_file.name)
else:
    env_df = pd.DataFrame()
    env_school_name = ""

if growth_file is not None:
    growth_df = read_csv_uploaded(growth_file)
    growth_df = std_cols(growth_df)
    growth_school_name = str(growth_file.name).split("_")[0] if "_" in growth_file.name else str(growth_file.name)
else:
    growth_df = pd.DataFrame()
    growth_school_name = ""

st.sidebar.markdown("---")
st.sidebar.header("📏 생육 측정 항목")
growth_metrics = st.sidebar.multiselect(
    "비교할 항목 선택 (기본 모두)",
    ["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
    default=["잎 수", "지상부 길이", "지하부 길이", "지상부 질량", "지하부 질량"],
)

# ─────────────────────────────────────────────
# 탭 구성
# ─────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📖 실험 개요", "🌡️ 환경 데이터", "📊 생육 결과"])

# ─────────────────────────────────────────────
# 공통 헬퍼
# ─────────────────────────────────────────────
def chart_base_layout(fig, title=""):
    fig.update_layout(
        font=dict(family="Malgun Gothic, Apple SD Gothic Neo, sans-serif"),
        title=dict(text=title, x=0.5),
        hovermode="x unified",
    )
    return fig


def numeric_dropna(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return s


# ─────────────────────────────────────────────
# Tab 1: 실험 개요
# ─────────────────────────────────────────────
with tab1:
    try:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🔍 연구 배경 및 목적")
            st.markdown("""
**극지식물(Eskimo potato 등)** 은 낮은 온도·건조·영양 poor 환경에서도 생존하는 식물로,
개체마다 생육 차이가 크게 나타납니다.

이 대시보드에서는 우리 학교 학생들이 키운 극지식물 개체의 **생육 결과 데이터**와
같은 기간 기록된 **환경 데이터(온도·습도·pH·EC)** 를 연결해 분석합니다.

**분석 목표**
1. 개체별 **생중량 분포**와 차이를 평균·표준편차·변동계수(CV)로 이해
2. 잎 수·지상부 길이와 생중량의 **관계(상관계수·산점도)** 확인
3. 생중량 **상위 25% vs 하위 25%** 개체 비교
            """)

            # 주석: 연구 배경 설명 끝

        with col2:
            st.subheader("📋 데이터 요약")

            # 개체 수
            n_growth = len(growth_df)
            if n_growth == 0:
                st.info("생육 CSV를 업로드하면 여기에 요약이 표시됩니다.")
            else:
                # 측정 기간
                time_col_exists = "time" in env_df.columns and env_df["time"].notna().any()
                if time_col_exists:
                    t_min = env_df["time"].min()
                    t_max = env_df["time"].max()
                    period_str = f"{t_min.strftime('%Y-%m-%d')} ~ {t_max.strftime('%Y-%m-%d')}"
                else:
                    period_str = "알 수 없음"

                # 측정 항목
                growth_cols = [c for c in ["잎 수", "지상부 길이", "지하부 길이",
                                            "생중량", "지상부 질량", "지하부 질량"]
                               if c in growth_df.columns]
                env_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                           if c in env_df.columns]

                summary_data = {
                    "항목": ["총 개체 수", "측정 기간", "생육 측정 항목", "환경 측정 항목"],
                    "값": [
                        f"{n_growth}개체",
                        period_str,
                        ", ".join(growth_cols) if growth_cols else "없음",
                        ", ".join(env_cols) if env_cols else "없음",
                    ],
                }
                st.table(pd.DataFrame(summary_data))

        st.markdown("---")
        st.subheader("📊 주요 지표 카드")

        m1, m2, m3, m4 = st.columns(4)

        # 총 개체수
        if n_growth > 0:
            m1.metric("총 개체수", f"{n_growth}개체")
        else:
            m1.metric("총 개체수", "–")

        # 평균 온도
        if "temperature" in env_df.columns:
            t_vals = numeric_dropna(env_df["temperature"])
            if len(t_vals) >= 2:
                m2.metric("평균 온도", f"{t_vals.mean():.1f} °C")
            else:
                m2.metric("평균 온도", "데이터 부족")
        else:
            m2.metric("평균 온도", "센서 데이터 없음")

        # 평균 습도
        if "humidity" in env_df.columns:
            h_vals = numeric_dropna(env_df["humidity"])
            if len(h_vals) >= 2:
                m3.metric("평균 습도", f"{h_vals.mean():.1f} %")
            else:
                m3.metric("평균 습도", "데이터 부족")
        else:
            m3.metric("평균 습도", "센서 데이터 없음")

        # 평균 생중량
        if "생중량" in growth_df.columns:
            b_vals = numeric_dropna(growth_df["생중량"])
            if len(b_vals) >= 2:
                m4.metric("평균 생중량", f"{b_vals.mean():.2f} g")
            else:
                m4.metric("평균 생중량", "데이터 부족")
        else:
            m4.metric("평균 생중량", "생중량 데이터 없음")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─────────────────────────────────────────────
# Tab 2: 환경 데이터
# ─────────────────────────────────────────────
with tab2:
    try:
        if env_df.empty:
            st.info("🌡️ 환경 CSV를 업로드하면 환경 데이터가 표시됩니다.")
        else:
            st.subheader("🌡️ 환경 센서 요약")

            env_num_cols = [c for c in ["temperature", "humidity", "ph", "ec"]
                           if c in env_df.columns]

            if env_num_cols:
                summary_rows = []
                for col in env_num_cols:
                    vals = numeric_dropna(env_df[col])
                    if len(vals) >= 2:
                        summary_rows.append({
                            "측정 항목": col,
                            "평균": round(vals.mean(), 2),
                            "최소": round(vals.min(), 2),
                            "최대": round(vals.max(), 2),
                        })
                env_summary = pd.DataFrame(summary_rows)
                st.table(env_summary)
            else:
                st.info("표시할 환경 수치 열이 없습니다.")

            st.markdown("---")
            st.subheader("📈 날짜별 시계열")

            if "time" in env_df.columns and env_df["time"].notna().any():
                env_df_sorted = env_df.sort_values("time").copy()
                env_df_sorted["date"] = env_df_sorted["time"].dt.date

                # 일별 평균
                daily = env_df_sorted.groupby("date")[env_num_cols].mean().reset_index()

                n_daily = len(daily)
                if n_daily < 2:
                    st.info("하루치 데이터만 있어 시계열 그래프를 그리기 어렵습니다.")
                else:
                    line_cols = [c for c in ["temperature", "humidity"] if c in env_num_cols]
                    if line_cols:
                        fig_ts = make_subplots(
                            rows=len(line_cols), cols=1,
                            subplot_titles=["온도 변화 (°C)", "습도 변화 (%)"],
                            shared_xaxes=True,
                            vertical_spacing=0.12,
                        )

                        for i, col in enumerate(line_cols):
                            fig_ts.add_trace(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily[col],
                                    mode="lines",
                                    name=col,
                                    line=dict(width=2),
                                ),
                                row=i + 1, col=1,
                            )

                        fig_ts = chart_base_layout(fig_ts, "날짜별 환경 변화")
                        fig_ts.update_yaxes(title_text="°C", row=1, col=1)
                        fig_ts.update_yaxes(title_text="%", row=2, col=1)
                        st.plotly_chart(fig_ts, width="stretch")

                    extra_cols = [c for c in ["ph", "ec"] if c in env_num_cols]
                    if extra_cols:
                        st.markdown("#### pH · EC 변화")
                        fig2 = make_subplots(
                            rows=len(extra_cols), cols=1,
                            subplot_titles=["pH 변화", "EC 변화 (mS/cm)"],
                            shared_xaxes=True,
                            vertical_spacing=0.12,
                        )
                        for i, col in enumerate(extra_cols):
                            fig2.add_trace(
                                go.Scatter(
                                    x=daily["date"],
                                    y=daily[col],
                                    mode="lines",
                                    name=col,
                                    line=dict(width=2),
                                ),
                                row=i + 1, col=1,
                            )
                        fig2 = chart_base_layout(fig2, "pH · EC 일별 변화")
                        st.plotly_chart(fig2, width="stretch")
                    else:
                        st.info("pH·EC 센서 데이터가 없어 pH·EC 그래프는 표시하지 않습니다.")
            else:
                st.info("시간 열이 제대로 파싱되지 않아 시계열 그래프를 그릴 수 없습니다.")

            with st.expander("📄 환경 데이터 원본 테이블"):
                if not env_df.empty:
                    st.dataframe(env_df, width="stretch")
                    env_csv = env_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 환경 CSV 다운로드",
                        data=env_csv,
                        file_name="환경데이터 다운로드.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 환경 데이터가 없습니다.")

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")


# ─────────────────────────────────────────────
# Tab 3: 생육 결과
# ─────────────────────────────────────────────
with tab3:
    try:
        if growth_df.empty:
            st.info("📊 생육 CSV를 업로드하면 생육 결과 그래프가 표시됩니다.")
        else:
            has_biomass = "생중량" in growth_df.columns

            # ── 핵심 결과 카드 ──
            st.subheader("🥇 핵심 결과")
            c1, c2, c3 = st.columns(3)

            if has_biomass:
                bio = numeric_dropna(growth_df["생중량"])
                if len(bio) >= 2:
                    mean_bio = bio.mean()
                    max_idx = bio.idxmax()
                    min_idx = bio.idxmin()

                    # 개체번호는 원본 값 그대로 표시 (소수점 제거)
                    id_col = "개체번호" if "개체번호" in growth_df.columns else None
                    max_id = growth_df.loc[max_idx, id_col] if id_col else "N/A"
                    min_id = growth_df.loc[min_idx, id_col] if id_col else "N/A"

                    # 정수 타입이면 int로, 문자열이면 그대로
                    if isinstance(max_id, float) and max_id == int(max_id):
                        max_id = int(max_id)
                    if isinstance(min_id, float) and min_id == int(min_id):
                        min_id = int(min_id)

                    c1.metric("평균 생중량", f"{mean_bio:.2f} g")
                    c2.metric("최대 생중량", f"{bio.max():.2f} g", delta=f"개체 {max_id}")
                    c3.metric("최소 생중량", f"{bio.min():.2f} g", delta=f"개체 {min_id}")
                else:
                    c1.metric("평균 생중량", "데이터 부족")
                    c2.metric("최대 생중량", "–")
                    c3.metric("최소 생중량", "–")
            else:
                c1.metric("평균 생중량", "생중량 열 없음")
                c2.metric("최대 생중량", "–")
                c3.metric("최소 생중량", "–")

            st.markdown("---")

            # ── 개체별 생육 비교 막대 그래프 ──
            st.subheader("📊 개체별 생육 비교")

            bar_metrics = [m for m in growth_metrics if m in growth_df.columns]

            for metric in bar_metrics:
                vals = numeric_dropna(growth_df[metric])
                if len(vals) < 2:
                    continue

                id_col = "개체번호" if "개체번호" in growth_df.columns else None

                if id_col:
                    ids = growth_df.loc[vals.index, id_col]
                    # 정수형 개체번호는 int로 표시
                    clean_ids = []
                    for v in ids:
                        if isinstance(v, float) and v == int(v):
                            clean_ids.append(int(v))
                        else:
                            clean_ids.append(v)
                else:
                    clean_ids = [f"개체{i+1}" for i in range(len(vals))]

                # 생중량이면 상위/하위 강조 색상
                is_bio = (metric == "생중량")
                if is_bio and has_biomass:
                    bio_all = numeric_dropna(growth_df["생중량"])
                    q75 = bio_all.quantile(0.75)
                    q25 = bio_all.quantile(0.25)

                fig_bar = go.Figure()
                colors = []
                for i, (idx, val) in enumerate(zip(clean_ids, vals)):
                    if is_bio:
                        if val >= q75:
                            colors.append("#2ecc71")  # 상위 25% - 초록
                        elif val <= q25:
                            colors.append("#e74c3c")  # 하위 25% - 빨강
                        else:
                            colors.append("#95a5a6")  # 중간 - 회색
                    else:
                        colors.append("#3498db")

                fig_bar.add_trace(go.Bar(
                    x=list(clean_ids),
                    y=list(vals),
                    marker_color=colors,
                    hovertemplate=f"{metric}<br>개체: %{{x}}<br>값: %{{y:.2f}}<extra></extra>",
                ))

                y_label = {"잎 수": "장", "지상부 길이": "mm",
                           "지하부 길이": "mm", "지상부 질량": "g",
                           "지하부 질량": "g"}.get(metric, "")
                fig_bar = chart_base_layout(fig_bar, f"개체별 {metric}")
                fig_bar.update_yaxes(title_text=y_label)
                fig_bar.update_xaxes(tickangle=45, title_text="개체번호" if id_col else "개체")

                if is_bio:
                    fig_bar.add_annotation(
                        text="■ 상위 25%   ■ 하위 25%   ■ 중간",
                        xref="paper", yref="paper",
                        x=0.5, y=-0.15, showarrow=False,
                        font=dict(size=11, color="gray"),
                    )

                st.plotly_chart(fig_bar, width="stretch")

            st.markdown("---")

            # ── 생중량 분포 (박스플롯) ──
            if has_biomass:
                bio_vals = numeric_dropna(growth_df["생중량"])
                if len(bio_vals) >= 4:
                    id_col = "개체번호" if "개체번호" in growth_df.columns else None
                    if id_col:
                        ids_for_box = growth_df.loc[bio_vals.index, id_col]
                        clean_ids_box = []
                        for v in ids_for_box:
                            if isinstance(v, float) and v == int(v):
                                clean_ids_box.append(int(v))
                            else:
                                clean_ids_box.append(v)
                    else:
                        clean_ids_box = [f"개체{i+1}" for i in range(len(bio_vals))]

                    fig_box = go.Figure()
                    fig_box.add_trace(go.Box(
                        y=list(bio_vals),
                        name="생중량",
                        fillcolor="#3498db",
                        marker=dict(color="#3498db"),
                        line=dict(color="black"),
                        boxmean=True,
                    ))
                    fig_box = chart_base_layout(fig_box, "생중량 분포 (박스플롯)")
                    fig_box.update_yaxes(title_text="생중량 (g)")
                    st.plotly_chart(fig_box, width="stretch")

                    # CV 계산
                    mean_v = bio_vals.mean()
                    std_v = bio_vals.std()
                    cv = (std_v / mean_v * 100) if mean_v != 0 else float("nan")
                    st.info(
                        f"**변동계수(CV)** = 표준편차 ÷ 평균 × 100 = {cv:.1f}%  "
                        f"→ CV가 클수록 개체 간 생육 차이가 큽니다."
                    )
                else:
                    st.info("생중량 데이터가 4개 미만이라 박스플롯을 그리기 어렵습니다.")
            else:
                st.info("생중량 열이 없어 생중량 분포 그래프를 그리지 않습니다.")

            st.markdown("---")

            # ── 상관관계 분석 (산점도 2개 + 추세선) ──
            st.subheader("🔗 상관관계 분석")

            bio_for_corr = numeric_dropna(growth_df["생중량"]) if has_biomass else None

            scatter_pairs = []
            if has_biomass:
                for x_col in ["잎 수", "지상부 길이"]:
                    if x_col in growth_df.columns:
                        scatter_pairs.append((x_col, "생중량"))

            if scatter_pairs:
                for x_col, y_col in scatter_pairs:
                    x_vals = numeric_dropna(growth_df[x_col])
                    y_vals = bio_for_corr

                    common_idx = x_vals.index.intersection(y_vals.index)
                    x_clean = x_vals.loc[common_idx]
                    y_clean = y_vals.loc[common_idx]

                    if len(x_clean) < 3:
                        st.info(f"{x_col} vs 생중량: 공통 데이터 {len(x_clean)}개로 산점도 생략")
                        continue

                    slope, intercept = np.polyfit(x_clean, y_clean, 1)
                    r, p = stats.pearsonr(x_clean, y_clean)

                    fig_sc = go.Figure()
                    fig_sc.add_trace(go.Scatter(
                        x=list(x_clean),
                        y=list(y_clean),
                        mode="markers",
                        name="데이터",
                        marker=dict(color="#3498db", size=8, opacity=0.7),
                        hovertemplate=f"{x_col} vs 생중량<br>{x_col}: %{{x:.1f}}<br>생중량: %{{y:.2f}}g<extra></extra>",
                    ))
                    # 추세선
                    x_line = np.array([x_clean.min(), x_clean.max()])
                    y_line = slope * x_line + intercept
                    fig_sc.add_trace(go.Scatter(
                        x=x_line,
                        y=y_line,
                        mode="lines",
                        name=f"추세선 (r={r:.2f})",
                        line=dict(color="#e74c3c", width=2, dash="dash"),
                    ))

                    fig_sc = chart_base_layout(fig_sc, f"{x_col} vs 생중량")
                    fig_sc.update_xaxes(title_text=x_col)
                    fig_sc.update_yaxes(title_text="생중량 (g)")

                    st.plotly_chart(fig_sc, width="stretch")

                    direction = "정" if r > 0 else "부"
                    strength = ("강한" if abs(r) >= 0.7 else
                                "중간" if abs(r) >= 0.4 else "약한")
                    st.markdown(
                        f"**{x_col} ↔ 생중량**<br>"
                        f"- 피어슨 상관계수 r = **{r:.3f}** (p = {p:.4f})<br>"
                        f"- {direction}의 {strength} 상관관계<br>"
                        f"- 추세선: 생중량 = {slope:.4f} × {x_col} + {intercept:.3f}<br>"
                        f"> **상관계수(r)** 는 −1 ~ 1 사이 값으로, "
                        f"1에 가까울수록 함께 증가, −1에 가까울수록 반대로 움직입니다."
                    )
                    st.markdown("---")
            else:
                st.info("잎 수 또는 지상부 길이 데이터와 생중량 데이터가 모두 있어야 산점도를 그립니다.")

            with st.expander("📄 생육 데이터 원본"):
                if not growth_df.empty:
                    st.dataframe(growth_df, width="stretch")
                    growth_csv = growth_df.to_csv(index=False).encode("utf-8-sig")
                    st.download_button(
                        "📥 생육 CSV 다운로드",
                        data=growth_csv,
                        file_name="생육결과 다운로드.csv",
                        mime="text/csv",
                    )
                else:
                    st.info("표시할 생육 데이터가 없습니다.")

            # ── 상위 25% vs 하위 25% 비교 ──
            if has_biomass:
                bio_all = numeric_dropna(growth_df["생중량"])
                if len(bio_all) >= 4:
                    q75 = bio_all.quantile(0.75)
                    q25 = bio_all.quantile(0.25)

                    top_idx = bio_all[bio_all >= q75].index
                    bot_idx = bio_all[bio_all <= q25].index

                    if len(top_idx) >= 2 and len(bot_idx) >= 2:
                        st.markdown("---")
                        st.subheader("🏆 생중량 상위 25% vs 하위 25% 비교")

                        t_mean = bio_all.loc[top_idx].mean()
                        b_mean = bio_all.loc[bot_idx].mean()
                        t_std = bio_all.loc[top_idx].std()
                        b_std = bio_all.loc[bot_idx].std()

                        comp_data = {
                            "구분": ["상위 25%", "하위 25%"],
                            "개체 수": [len(top_idx), len(bot_idx)],
                            "평균 생중량 (g)": [round(t_mean, 2), round(b_mean, 2)],
                            "표준편차 (g)": [round(t_std, 2), round(b_std, 2)],
                        }
                        st.table(pd.DataFrame(comp_data))

                        st.info(
                            f"상위 25% 평균({t_mean:.2f}g)이 하위 25% 평균({b_mean:.2f}g)의 "
                            f"약 {t_mean/b_mean:.1f}배입니다."
                        )

    except Exception as e:
        st.warning(f"⚠️ 이 부분은 데이터 형식 때문에 표시하지 못했어요: {e}")
