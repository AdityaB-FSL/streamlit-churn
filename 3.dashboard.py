import streamlit as st
import pickle
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go

# ========== Data & Model Loaders ==========

def load_data():
    return pd.read_csv("data/baseline_model.csv")


# ========== Streamlit UI ==========


col1, col2 = st.columns([5,1])
with col1:
    st.title("📊 Dashboard for Churn Analysis")
    st.write("This dashboard shows how each feature affects customer churn.")
with col2:
    st.image("assets/logo.jpg",width=150)
st.divider()


data = load_data()
if 'subscription_status' not in data.columns and 'churn' in data.columns:
    data['subscription_status'] = data['churn'].map({0: 'No Churn', 1: 'Churn'})

feature = st.selectbox("🔎 Select Feature to Analyze", [col for col in data.columns if col not in ['churn', 'subscription_status']])
analysis_type = st.radio("📊 Select Analysis Type", ["Univariate", "Bivariate"], horizontal=True)
st.markdown("---")

col1, col2, col3 = st.columns(3)
if pd.api.types.is_numeric_dtype(data[feature]):
    with st.container(border=True):
        col1.metric("Mean", f"{data[feature].mean():.2f}")
        col2.metric("Median", f"{data[feature].median():.2f}")
        col3.metric("Std Dev", f"{data[feature].std():.2f}")
else:
    with st.container(border=True):
        col1.metric("Unique Values", data[feature].nunique())
        top_cat = data[feature].value_counts().idxmax()
        col2.metric("Most Common", str(top_cat))
        col3.metric("Total Count", len(data))

st.markdown("---")

if analysis_type == "Univariate":
    st.subheader("📌 Univariate Distribution")
    with st.container(border=True):
        if pd.api.types.is_numeric_dtype(data[feature]):
            fig = px.histogram(data, x=feature, nbins=30, title=f"{feature} Distribution", marginal="violin")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(data, y=feature, title=f"{feature} Boxplot")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            counts = data[feature].value_counts().reset_index()
            counts.columns = [feature, "count"]
            fig = px.bar(counts, x=feature, y="count", title=f"{feature} Count")
            st.plotly_chart(fig, use_container_width=True)

else:
    st.subheader("🔁 Relationship with Churn")
    with st.container(border=True):
        if pd.api.types.is_numeric_dtype(data[feature]):
            fig = px.histogram(data, x=feature, color='subscription_status', barmode='overlay', opacity=0.7, title=f"{feature} by Churn")
            st.plotly_chart(fig, use_container_width=True)

            fig2 = px.box(data, x='subscription_status', y=feature, color='subscription_status', title=f"{feature} vs Churn Category")
            st.plotly_chart(fig2, use_container_width=True)
        else:
            fig = px.histogram(data, x=feature, color='subscription_status', barmode='group', title=f"{feature} vs Churn")
            st.plotly_chart(fig, use_container_width=True)

            churn_table = pd.crosstab(data[feature], data['subscription_status'], normalize='index') * 100
            st.markdown("#### 📊 Churn Rate by Category")
            st.dataframe(churn_table.style.format("{:.1f}%").background_gradient(axis=1, cmap="RdYlGn_r"))

st.markdown("---")