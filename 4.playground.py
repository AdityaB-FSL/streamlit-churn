import streamlit as st
import pickle
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go

# ========== Data & Model Loaders ==========

def load_data():
    return pd.read_csv("data/baseline_model.csv")

def load_xgb_model():
    with open('model/model_2.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# ========== Preprocessing ==========

def preprocess(user_input):
    df = pd.DataFrame([user_input])
    df['subscription_type'] = df['subscription_type'].map({'Espresso': 0, 'Digital': 1, 'Digital+Print': 2})
    df['plan_type'] = df['plan_type'].map({'Monthly': 0, 'Annual': 1})
    df['auto_renew'] = df['auto_renew'].map({'Yes': 1, 'No': 0})
    df['discount_used_last_renewal'] = df['discount_used_last_renewal'].map({'Yes': 1, 'No': 0})
    df['downgrade_history'] = df['downgrade_history'].map({'Yes': 1, 'No': 0})
    df['previous_renewal_status'] = df['previous_renewal_status'].map({'Auto': 1, 'Manual': 0})
    df['signup_source'] = df['signup_source'].map({'Web': 0, 'Mobile App': 0, 'Referral': 1})

    df = pd.get_dummies(df, columns=['region', 'most_read_category', 'primary_device', 'payment_method', 'last_campaign_engaged'])
    df = df.fillna(0)


    MODEL_FEATURES = ['subscription_type', 'plan_type', 'auto_renew',
       'avg_articles_per_week', 'days_since_last_login',
       'support_tickets_last_90d', 'discount_used_last_renewal',
       'email_open_rate', 'time_spent_per_session_mins',
       'completion_rate', 'article_skips_per_week',
       'previous_renewal_status', 'campaign_ctr', 'nps_score',
       'sentiment_score', 'csat_score', 'customer_age', 'signup_source',
       'downgrade_history', 'tenure_days', 'region_Asia', 'region_Europe',
       'region_North America', 'region_Others', 'most_read_Culture',
       'most_read_Environment', 'most_read_Finance', 'most_read_Politics',
       'most_read_Technology', 'primary_device_Desktop',
       'primary_device_Mobile', 'primary_device_Tablet',
       'payment_method_Credit Card', 'payment_method_Debit Card',
       'payment_method_PayPal', 'last_campaign_engaged_Newsletter Promo',
       'last_campaign_engaged_Retention Offer',
       'last_campaign_engaged_Survey']
    
    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = 0

    return df[MODEL_FEATURES]



# ========== Streamlit UI ==========


col1, col2 = st.columns([5,1])
with col1:
    st.title("🚀 Playground")
    st.markdown("Here you can predict whether a customer will churn based on their engagement, behavior, and subscription data.")
with col2:
    st.image("assets/logo.jpg",width=150)
st.divider()


st.subheader("📋 User Input Options")
col1, col2 = st.columns(2)
with col1:
    with st.expander("📦 Subscription Details", expanded=True):
        subscription_type = st.selectbox('Subscription Plan Type', ['Espresso', 'Digital', 'Digital+Print'])
        plan_type = st.selectbox('Plan Type', ['Monthly', 'Annual'])
        signup_source = st.selectbox('Signup Source', ['Web', 'Mobile App', 'Referral'])
        auto_renew = st.toggle("Was Auto-renew Enabled on Subscription?")
        discount_used_last_renewal = st.toggle("Was Discount Used at Last Renewal?")
        previous_renewal_status = st.toggle("Was Subscription Previously Renewed?")
        downgrade_history = st.toggle("Was Subscription Downgraded Before?")
with col2:
    with st.expander("🌍 User Profile", expanded=True):
        region = st.selectbox('Region', ['North America', 'Europe', 'Asia', 'Other'])
        primary_device = st.selectbox('Primary Device', ['Tablet', 'Mobile', 'Desktop'])
        payment_method = st.selectbox('Payment Method', ['Credit Card', 'Debit Card', 'PayPal'])
        most_read_category = st.selectbox('Most Read Category', ['Technology', 'Business', 'Science', 'Health', 'Politics', 'Entertainment', 'Culture'])
        last_campaign_engaged = st.selectbox('Last Campaign Engaged', ['Newsletter Promo', 'Retention Offer', 'Survey'])

if "condition" not in st.session_state:
    st.session_state["condition"] = True


col3, col4 = st.columns(2)
with col3:
    with st.expander("📊 Engagement Metrics", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            customer_age = st.slider('Customer Age', 18, 100, 25)
            avg_articles_per_week = st.slider('Avg Articles/Week', 0.0, 9.0, 0.0, 0.1)
            article_skips_per_week = st.slider('Article Skips/Week', 0, 10, 0)
            days_since_last_login = st.slider('Days Since Last Login', 0, 100, 0)
        with col2:
            support_tickets_last_90d = st.slider('Support Tickets (Last 90 Days)', 0, 10, 0)
            email_open_rate = st.slider('Email Open Rate', 0.00, 1.00, 0.00, 0.01)
            time_spent_per_session_mins = st.slider('Time/Session (mins)', 0.0, 30.0, 0.0, 0.1)
            tenure_days = st.slider('Tenure (Days)', 0, 1825, 25)

with col4:
    with st.expander("📈 Engagement Scores", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            completion_rate = st.slider('Completion Rate', 0.00, 1.00, 0.00, 0.01)
            campaign_ctr = st.slider('Campaign CTR', 0.00, 1.00, 0.00, 0.01)
        with col2:
            nps_score = st.slider('NPS Score', -100, 100, 0)
            sentiment_score = st.slider('Sentiment Score', -1.5, 1.5, 0.0, 0.1)
            csat_score = st.slider('CSAT Score (1-5)', 1, 5, 3)

col5, col6 = st.columns([1, 1])
predict_button = col5.button("🔍 Predict", use_container_width=True)
clear_button = col6.button("🗑️ Clear Inputs", use_container_width=True)

user_input = {
    'subscription_type': subscription_type,
    'plan_type': plan_type,
    'primary_device': primary_device,
    'region': region,
    'most_read_category': most_read_category,
    'last_campaign_engaged': last_campaign_engaged,
    'payment_method': payment_method,
    'signup_source': signup_source,
    'customer_age': customer_age,
    'avg_articles_per_week': avg_articles_per_week,
    'article_skips_per_week': article_skips_per_week,
    'days_since_last_login': days_since_last_login,
    'support_tickets_last_90d': support_tickets_last_90d,
    'email_open_rate': email_open_rate,
    'time_spent_per_session_mins': time_spent_per_session_mins,
    'tenure_days': tenure_days,
    'completion_rate': completion_rate,
    'campaign_ctr': campaign_ctr,
    'nps_score': nps_score,
    'sentiment_score': sentiment_score,
    'csat_score': csat_score,
    'discount_used_last_renewal': 'Yes' if discount_used_last_renewal else 'No',
    'auto_renew': 'Yes' if auto_renew else 'No',
    'previous_renewal_status': 'Auto' if previous_renewal_status else 'Manual',
    'downgrade_history': 'Yes' if downgrade_history else 'No'
}


if clear_button:
    st.rerun()

if predict_button:
    df = preprocess(user_input)
    model = load_xgb_model()
    prediction = model.predict(df)[0]
    proba = model.predict_proba(df)[0][1]

    result = "Churn" if prediction == 1 else "No Churn"
    st.success(f"📊 **Prediction:** {result}")
    st.info(f"🧠 Confidence: **{proba * 100:.2f}%** for Churn")

    st.subheader("🔎 Feature Importance")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(df)

    # --- Waterfall Plot ---
    row = shap_values[0]
    shap_impact = row.values
    features = row.feature_names
    base_val = row.base_values

    top_idx = np.argsort(np.abs(shap_impact))[-15:]
    top_features = [features[i] for i in top_idx]
    top_shap = shap_impact[top_idx]

    fig_waterfall = go.Figure(go.Waterfall(
        orientation="h",
        measure=["relative"] * len(top_features),
        x=top_shap,
        y=top_features,
        text=[f"{v:.3f}" for v in top_shap],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "red"}},
        increasing={"marker": {"color": "green"}},
    ))

    fig_waterfall.update_layout(
        title="",
        xaxis_title="SHAP Value Impact",
        yaxis_title="Feature",
        waterfallgap=0.4
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

