import io
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Smart Learning Path Recommender", page_icon="🎓", layout="wide")

# ---------------- UTILITIES ----------------
LEVEL_ORDER = ["Beginner", "Intermediate", "Expert"]

def _to_numeric(s):
    return pd.to_numeric(s, errors="coerce")

def clean_level(s):
    if pd.isna(s):
        return np.nan
    s = str(s).strip()
    mapping = {
        "Beginner Level": "Beginner",
        "Intermediate Level": "Intermediate",
        "Expert Level": "Expert",
        "All Levels": "Intermediate",  # map All Levels to middle for ordering
        "All Level": "Intermediate"
    }
    return mapping.get(s, s)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data(path="udemy_online_education_courses_dataset.csv") -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic cleaning
    if "course_title" in df.columns:
        df.dropna(subset=["course_title"], inplace=True)
    df.dropna(subset=["subject", "level"], inplace=True)
    df["level"] = df["level"].map(clean_level)
    df.dropna(subset=["level"], inplace=True)
    df.drop_duplicates(inplace=True)

    # Coerce numeric fields
    for col in ["price", "num_subscribers", "num_reviews", "num_lectures", "content_duration"]:
        if col in df.columns:
            df[col] = _to_numeric(df[col])

    # Parse timestamp
    if "published_timestamp" in df.columns:
        df["published_timestamp"] = pd.to_datetime(df["published_timestamp"], errors="coerce")

    # Composite score (engagement proxy)
    # weights can be tuned; keep bounded even with NAs
    df["course_score"] = (
        df["num_subscribers"].fillna(0) * 0.5 +
        df["num_reviews"].fillna(0) * 0.3 +
        df["content_duration"].fillna(0) * 0.2
    )

    # Order level categorically
    df["level"] = pd.Categorical(df["level"], categories=LEVEL_ORDER, ordered=True)

    # Ensure essential columns exist
    required_cols = ["course_title", "subject", "level", "price", "num_subscribers",
                     "num_reviews", "content_duration", "course_score"]
    for c in required_cols:
        if c not in df.columns:
            df[c] = np.nan

    return df

df = load_data()

# ---------------- GLOBAL STATE ----------------
if "my_path" not in st.session_state:
    st.session_state["my_path"] = []

# ---------------- SIDEBAR ----------------
st.sidebar.title("Smart Learning Path Recommender")
# Global filters
subjects = ["All"] + sorted(df["subject"].dropna().unique().tolist())
levels = ["All"] + LEVEL_ORDER

st.sidebar.subheader("🔎 Global Filters")
selected_subject = st.sidebar.selectbox("Filter by Subject", subjects, index=0)
selected_level = st.sidebar.selectbox("Filter by Level", levels, index=0)

# Price range filter (global)
price_min = float(np.nanmin(df["price"])) if df["price"].notna().any() else 0.0
price_max = float(np.nanmax(df["price"])) if df["price"].notna().any() else 100.0
price_low, price_high = st.sidebar.slider(
    "Filter by Price Range ($)",
    min_value=float(np.floor(price_min)),
    max_value=float(np.ceil(price_max if price_max > price_min else price_min + 1)),
    value=(float(np.floor(price_min)), float(np.ceil(price_max if price_max > price_min else price_min + 1)))
)

# Build filtered_df once (used by all tabs)
filtered_df = df.copy()
if selected_subject != "All":
    filtered_df = filtered_df[filtered_df["subject"] == selected_subject]
if selected_level != "All":
    filtered_df = filtered_df[filtered_df["level"] == selected_level]
filtered_df = filtered_df[(filtered_df["price"] >= price_low) & (filtered_df["price"] <= price_high)]

# ---------------- PRECOMPUTE: TF-IDF & COSINE ----------------
@st.cache_resource
def build_tfidf_matrix(c_titles: pd.Series):
    tfidf = TfidfVectorizer(stop_words="english")
    matrix = tfidf.fit_transform(c_titles.fillna(""))
    return tfidf, matrix

tfidf, tfidf_matrix = build_tfidf_matrix(df["course_title"])
title_to_index = pd.Series(df.index, index=df["course_title"]).drop_duplicates()

def hybrid_recommend(reference_title: str, n: int = 8, subject=None, level=None, budget=None):
    """Content-based (title) + profile filtering."""
    if reference_title not in title_to_index.index:
        return pd.DataFrame({"course_title": ["Course not found in dataset"]})
    ref_idx = title_to_index[reference_title]
    sims = cosine_similarity(tfidf_matrix[ref_idx], tfidf_matrix).flatten()
    # Build dataframe for ranking
    base = df.copy()
    base["sim_score"] = sims

    # Optional profile filters
    if subject:
        base = base[base["subject"] == subject]
    if level:
        base = base[base["level"] == level]
    if budget is not None:
        base = base[base["price"] <= budget]

    # Drop self, sort by similarity then by score
    base = base[base.index != ref_idx]
    base = base.sort_values(["sim_score", "course_score"], ascending=[False, False]).head(n)

    cols = ["course_title", "subject", "level", "price", "num_subscribers", "num_reviews", "content_duration", "sim_score"]
    return base[cols]

def get_learning_path(subject: str, budget: float | None = None, level_filter: str | None = None):
    """Return ordered path within subject; optionally constrained by level and budget."""
    path = df[df["subject"] == subject].copy()
    if budget is not None:
        path = path[path["price"] <= budget]
    if level_filter in LEVEL_ORDER:
        path = path[path["level"] == level_filter]
    path = path.sort_values(["level", "course_score"], ascending=[True, False])
    return path[["course_title", "level", "price", "content_duration", "num_subscribers"]].reset_index(drop=True)

# ---------------- CLUSTERING (on full df, reused per-filter) ----------------
@st.cache_resource
def add_clusters(_df: pd.DataFrame, n_clusters: int = 4):
    work = _df.copy()
    features = work[["price", "content_duration", "num_subscribers", "num_reviews", "course_score"]].fillna(0.0)
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    labels = km.fit_predict(X)
    work["cluster"] = labels
    return work, km

df_clustered, km_model = add_clusters(df)

# Merge cluster labels back into filtered_df
if "cluster" in df_clustered.columns:
    filtered_df = filtered_df.merge(df_clustered[["course_title", "cluster"]], on="course_title", how="left")

# ---------------- HEADER KPIs ----------------
st.subheader("📊 Platform Summary")
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Courses (filtered)", int(filtered_df.shape[0]))
popular_subject = (filtered_df["subject"].mode().iloc[0] if not filtered_df["subject"].dropna().empty else "—")
k2.metric("Most Popular Subject", popular_subject)
avg_price = filtered_df["price"].mean() if filtered_df["price"].notna().any() else 0.0
k3.metric("Avg. Price (filtered)", f"${avg_price:,.2f}")
k4.metric("Avg. Duration (hrs)", f"{filtered_df['content_duration'].mean():.1f}" if filtered_df["content_duration"].notna().any() else "—")

st.caption(f"Filters → **Subject:** {selected_subject} | **Level:** {selected_level} | **Price:** ${price_low:,.0f}–${price_high:,.0f}")

# ---------------- TABS ----------------
tab_overview, tab_recs, tab2, tab_insights, tab_clusters, tab_pricing = st.tabs([
    "Overview", "Recommendations", "Learners", "Course Insights", "Clustering & Trends", "Pricing & Popularity"
])

# ===== TAB: OVERVIEW =====
with tab_overview:
    c1, c2 = st.columns([1, 1])

    with c1:
        fig_hist = px.histogram(
            filtered_df,
            x="subject",
            color="level",
            barmode="group",
            title="Course Distribution by Subject & Level"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Published timeline if available
        if "published_timestamp" in filtered_df.columns and filtered_df["published_timestamp"].notna().any():
            by_month = (
                filtered_df.set_index("published_timestamp")
                .resample("M")["course_title"]
                .count()
                .rename("courses_added")
                .reset_index()
            )
            fig_time = px.line(by_month, x="published_timestamp", y="courses_added", title="Courses Published Over Time")
            st.plotly_chart(fig_time, use_container_width=True)

    with c2:
        # Correlation Heatmap
        corr_cols = ["price", "content_duration", "num_subscribers", "num_reviews", "course_score"]
        corr_df = filtered_df[corr_cols].dropna()
        if not corr_df.empty:
            fig_corr, ax = plt.subplots()
            sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            ax.set_title("Feature Correlations (filtered)")
            st.pyplot(fig_corr)
        else:
            st.info("Not enough data to compute correlations on the filtered selection.")

# ===== TAB: RECOMMENDATIONS =====
with tab_recs:
    st.header("🎯 Personalized Recommendations")
    colA, colB = st.columns([1, 1])

    # ---- LEFT COLUMN ----
    with colA:
        st.subheader("Your Profile")
        budget = st.slider(
            "Max Budget ($)",
            min_value=float(np.floor(price_min)),
            max_value=float(np.ceil(price_max if price_max > price_min else price_min + 1)),
            value=float(np.ceil(min(price_high, max(price_min, avg_price if avg_price > 0 else price_max))))
        )
        st.caption("We’ll prefer courses at or under your budget.")

        # Reference Course
        if filtered_df.empty:
            st.warning("No courses match current filters. Loosen filters to see reference courses.")
            ref_courses = sorted(df["course_title"].unique().tolist())
        else:
            ref_courses = sorted(filtered_df["course_title"].unique().tolist())

        reference_title = st.selectbox("Pick a reference course", ref_courses)

        n_recs = st.slider("Number of recommendations", 3, 15, 8)

        # Compute recommendations
        profile_subject = None if selected_subject == "All" else selected_subject
        profile_level = None if selected_level == "All" else selected_level
        recs = hybrid_recommend(reference_title, n=n_recs, subject=profile_subject, level=profile_level, budget=budget)

        st.subheader("Recommended Courses")
        if recs.empty or "Course not found" in str(recs.iloc[0, 0]):
            st.info("No recommendations could be generated for the current filters.")
        else:
            # Card layout instead of dataframe
            for idx, row in recs.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div style="padding:12px; margin-bottom:10px; border-radius:12px; 
                                    border:1px solid #ddd; background:#f9f9f9;">
                            <h4 style="margin:0;">{row['course_title']}</h4>
                            <p style="margin:2px 0;"><b>Subject:</b> {row['subject']}</p>
                            <p style="margin:2px 0;"><b>Level:</b> {row['level']}</p>
                            <p style="margin:2px 0;"><b>Price:</b> ${row['price']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )


    # ---- RIGHT COLUMN ----
    with colB:
       
        st.subheader("Suggested Learning Path")
        if selected_subject != "All":
            chosen_subject = selected_subject
            st.caption(f"Using global subject filter: **{chosen_subject}**")
        else:
            chosen_subject = st.selectbox("Choose a subject for path", sorted(df["subject"].unique()))

        lp = get_learning_path(chosen_subject, budget=budget, level_filter=(None if selected_level == "All" else selected_level))
        if lp.empty:
            st.info("No courses match the path settings. Try adjusting budget/filters.")
        else:
            top_lp = lp.head(3)
            for idx, row in top_lp.iterrows():
                with st.container():
                    st.markdown(
                        f"""
                        <div style="padding:12px; margin-bottom:10px; border-radius:12px; 
                                    border:1px solid #ddd; background:#eef6ff;">
                            <h4 style="margin:0;">{row['course_title']}</h4>
                            <p style="margin:2px 0;"><b>Level:</b> {row['level']}</p>
                            <p style="margin:2px 0;"><b>Price:</b> ${row['price']}</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
# ===== LEARNERS =====

# Map sidebar level filter to numeric skill
skill_map = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
st.session_state["learner_skill"] = skill_map.get(selected_level, 1)

# --------- LEARNER PROFILE DASHBOARD ---------
with tab2:
    st.header("🎓 Learner Profile Dashboard")
    st.markdown("Customize your preferences and see your personalized profile summary with skill gap insights.")

    # ---------------- QUICK PICKS: TOP 3 + RADAR + TIME ESTIMATE ----------------
    st.subheader("🎯 Quick Picks For You (Top 3)")

    # ---- Inputs for this block ----
    col_q1, col_q2 = st.columns([1, 1])

    with col_q1:
        # Learner skill (used to choose next target level)
        learner_skill = st.select_slider(
            "Your Skill Level",
            options=LEVEL_ORDER,
            value=(selected_level if selected_level in LEVEL_ORDER else "Beginner"),
            help="We'll target the next level up for a challenging-but-feasible path."
        )

    with col_q2:
        # Time availability → hours/week mapping
        time_choice = st.selectbox(
            "⏰ Time Availability (per week)",
            options=["<2 hrs", "2-5 hrs", "5-10 hrs", "10+ hrs"],
            index=2,
            help="Used to estimate completion time for your top 3 picks."
        )

    # Helper maps
    level_to_num = {"Beginner": 1, "Intermediate": 2, "Expert": 3}
    num_to_level = {v: k for k, v in level_to_num.items()}

    time_to_hours = {
        "<2 hrs": 1.5,
        "2-5 hrs": 4.0,
        "5-10 hrs": 8.0,
        "10+ hrs": 12.0
    }
    hours_per_week = time_to_hours.get(time_choice, 6.0)

    # Target level = next level up (Beginner->Intermediate, Intermediate->Expert, Expert->Expert)
    cur_level_num = level_to_num.get(learner_skill, 1)
    target_level_num = min(cur_level_num + 1, 3)
    target_level = num_to_level[target_level_num]

    # ---- Build candidate set (subject + target level emphasis only) ----
    if selected_subject != "All":
        local_subject = selected_subject
    else:
        local_subject = st.selectbox("Preferred Subject Area", sorted(df["subject"].dropna().unique()))

    candidates = df.copy()
    candidates = candidates[
        (candidates["subject"] == local_subject) &
        (candidates["level"].notna())
    ]

    # Pick top 3 candidates (just take first 3 or random)
    if not candidates.empty:
        top3 = candidates.sample(min(3, len(candidates)), random_state=42)
    else:
        top3 = pd.DataFrame()

    # ---- Output: Top 3 recommended courses ----
    if top3.empty:
        st.warning("No courses match your current subject and level. Try adjusting filters.")
    else:
        show_cols = ["course_title", "subject", "level", "price", "content_duration",
                     "num_subscribers", "num_reviews"]
        st.dataframe(top3[show_cols].reset_index(drop=True), use_container_width=True)

        # Add to path button
        c_add = st.columns(3)
        if c_add[0].button("Add Top 3 to My Path", key="add_top3_path"):
            for t in top3["course_title"]:
                if t not in st.session_state["my_path"]:
                    st.session_state["my_path"].append(t)
            st.success("Top 3 courses added to your learning path.")

    # ---- Skill Gap Radar: learner vs target level ----
    st.markdown("### 📊 Skill Gap Radar (Your Level vs Target)")
    radar_categories = ["Beginner", "Intermediate", "Expert"]
    learner_r = [1 if learner_skill == c else 0 for c in radar_categories]
    target_r  = [1 if target_level  == c else 0 for c in radar_categories]

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=learner_r, theta=radar_categories, fill="toself", name="Your Current Level"
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=target_r, theta=radar_categories, fill="toself", name=f"Target Level ({target_level})"
    ))
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=False, range=[0, 1])),
        showlegend=True
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # ---- Time Estimate: total hours of Top 3 vs availability ----
    st.markdown("### 🕒 Expected Completion Time vs Availability")

    if top3.empty:
        st.info("Set filters to get top 3 picks and see timing estimates.")
    else:
        total_hours = float(top3["content_duration"].fillna(0).sum())
        weeks_needed = np.ceil(total_hours / max(hours_per_week, 0.5))  # avoid divide by zero
        months_needed = weeks_needed / 4.345

        c_time1, c_time2, c_time3 = st.columns([1, 1, 2])
        c_time1.metric("Total Hours (Top 3)", f"{total_hours:.1f} h")
        c_time2.metric("Your Weekly Capacity", f"{hours_per_week:.1f} h / week")
        c_time3.metric("Estimated Duration", f"{int(weeks_needed)} weeks (~{months_needed:.1f} months)")

# ===== TAB: COURSE INSIGHTS =====
with tab_insights:
    st.header("📈 Course Insights")
    c1, c2 = st.columns(2)
    with c1:
        fig_s1 = px.scatter(
            filtered_df,
            x="price",
            y="course_score",
            color="subject" if selected_subject == "All" else None,
            hover_data=["course_title", "level", "num_subscribers", "num_reviews"],
            title="Price vs Course Score"
        )
        st.plotly_chart(fig_s1, use_container_width=True)

        # Value score = score / price (avoid divide-by-zero)
        work = filtered_df.copy()
        work["value_score"] = work["course_score"] / work["price"].replace(0, np.nan)
        top_value = work.sort_values("value_score", ascending=False).head(10).dropna(subset=["value_score"])
        if not top_value.empty:
            fig_bar_value = px.bar(
                top_value,
                x="course_title",
                y="value_score",
                color="subject" if selected_subject == "All" else None,
                title="Top 10 Best Value Courses (Score per $)"
            )
            st.plotly_chart(fig_bar_value, use_container_width=True)
        else:
            st.info("Not enough data to compute value scores.")

    with c2:
        # Popularity vs Engagement
        fig_s2 = px.scatter(
            filtered_df,
            x="num_reviews",
            y="num_subscribers",
            color="level" if selected_level == "All" else None,
            hover_data=["course_title", "subject"],
            title="Subscribers vs Reviews (Popularity & Engagement)"
        )
        st.plotly_chart(fig_s2, use_container_width=True)

        # Top subscribed
        top_sub = filtered_df.sort_values("num_subscribers", ascending=False).head(10)
        fig_bar_pop = px.bar(
            top_sub,
            x="course_title",
            y="num_subscribers",
            color="subject" if selected_subject == "All" else None,
            title="Top 10 Courses by Subscribers"
        )
        st.plotly_chart(fig_bar_pop, use_container_width=True)

# ===== TAB: CLUSTERING & TRENDS =====
with tab_clusters:
    st.header("🔬 Clustering & Trends")
    if "cluster" not in filtered_df.columns or filtered_df["cluster"].isna().all():
        st.info("Clusters unavailable for current selection.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            fig_c1 = px.scatter(
                filtered_df,
                x="price",
                y="course_score",
                color=filtered_df["cluster"].astype(str),
                hover_data=["course_title", "subject", "level"],
                title="Clusters: Price vs Course Score"
            )
            st.plotly_chart(fig_c1, use_container_width=True)

        with c2:
            # Cluster profiles
            prof = filtered_df.groupby("cluster").agg(
                avg_price=("price", "mean"),
                avg_duration=("content_duration", "mean"),
                avg_subscribers=("num_subscribers", "mean"),
                avg_reviews=("num_reviews", "mean"),
                avg_score=("course_score", "mean"),
                count=("course_title", "count")
            ).reset_index()
            st.dataframe(prof)

            # Simple narrative
            for _, r in prof.iterrows():
                st.caption(
                    f"Cluster {int(r['cluster'])}: {int(r['count'])} courses — "
                    f"avg price ${r['avg_price']:.0f}, duration {r['avg_duration']:.1f}h, "
                    f"subs {r['avg_subscribers']:.0f}, reviews {r['avg_reviews']:.0f}, score {r['avg_score']:.0f}"
                )

# ===== TAB: PRICING & POPULARITY =====
with tab_pricing:
    st.header("💰 Pricing & Popularity")
    c1, c2 = st.columns(2)
    with c1:
        fig_price_hist = px.histogram(filtered_df, x="price", nbins=40, title="Price Distribution")
        st.plotly_chart(fig_price_hist, use_container_width=True)

        fig_subj_avg = px.bar(
            filtered_df.groupby("subject", as_index=False)["price"].mean(),
            x="subject",
            y="price",
            title="Average Price by Subject"
        )
        st.plotly_chart(fig_subj_avg, use_container_width=True)

    with c2:
        fig_demand = px.scatter(
        filtered_df,
        x="price",
        y="num_subscribers",
        color="subject",
        hover_data=["course_title"],
        title="Price vs Subscribers (Demand Sensitivity)"
        )
        st.plotly_chart(fig_demand, use_container_width=True)

        fig_dur_rev = px.scatter(
            filtered_df,
            x="content_duration",
            y="num_reviews",
            color="subject" if selected_subject == "All" else None,
            title="Content Duration vs Reviews (Engagement)"
        )
        st.plotly_chart(fig_dur_rev, use_container_width=True)

# ---------------- DOWNLOADS ----------------
st.markdown("### 📥 Downloads")
d1, d2, d3 = st.columns(3)
with d1:
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    st.download_button("Download Full Dataset (CSV)", data=buf.getvalue(), file_name="udemy_courses_full.csv", mime="text/csv")
with d2:
    buf2 = io.StringIO()
    filtered_df.to_csv(buf2, index=False)
    st.download_button("Download Filtered Dataset (CSV)", data=buf2.getvalue(), file_name="udemy_courses_filtered.csv", mime="text/csv")
with d3:
    if st.session_state["my_path"]:
        path_df = pd.DataFrame({"course_title": st.session_state["my_path"]})
        buf3 = io.StringIO()
        path_df.to_csv(buf3, index=False)
        st.download_button("Download My Learning Path (CSV)", data=buf3.getvalue(), file_name="my_learning_path.csv", mime="text/csv")
    else:
        st.caption("Add courses to your path from the Recommendations tab to enable download.")
