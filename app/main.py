import streamlit as st
from model_utils import predict
from news_utils import fetch_related_news

st.set_page_config(page_title="NEO-REFUTE", layout="centered")
st.title("⚡ NEO-REFUTE: Real-Time Fake News Detection")

claim = st.text_input("Enter a news headline or claim:")

if st.button("Check Now"):
    if claim.strip() == "":
        st.warning("⚠️ Please enter a claim first.")
    else:
        st.subheader("📰 Claim")
        st.write(claim)

        # Predict on claim
        claim_pred = predict(claim)
        st.write("🤖 Model Prediction:", f"**{claim_pred}**")

        # Fetch related news
        related = fetch_related_news(claim)
        if related:
            st.subheader("🔎 Related Articles")
            for art in related:
                ref_pred = predict(art)
                st.write(f"- {art} → **{ref_pred}**")
        else:
            st.warning("No related articles found online!")
