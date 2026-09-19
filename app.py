"""
PhishGuard -- Streamlit frontend.

Run from the project root:
    streamlit run app.py
"""
from __future__ import annotations
import streamlit as st

from scripts.predict import predict

st.set_page_config(page_title="PhishGuard", page_icon="🛡️", layout="centered")

st.title("🛡️ PhishGuard")
st.caption("Dual-encoder phishing detector -- ModernBERT (email text) + CANINE (URL) with gated fusion")

st.markdown(
    "Paste an email body and/or a URL below. You can provide either one, or both -- "
    "the model adapts automatically to whichever information is available."
)

with st.form("phish_form"):
    email_text = st.text_area(
        "Email body (optional)",
        height=180,
        placeholder="Paste the email text here...",
    )
    url_text = st.text_input(
        "URL (optional)",
        placeholder="https://example.com/login",
    )
    submitted = st.form_submit_button("Analyze", use_container_width=True)

if submitted:
    if not email_text.strip() and not url_text.strip():
        st.warning("Please provide at least an email body or a URL.")
    else:
        with st.spinner("Analyzing..."):
            try:
                result = predict(text=email_text, url=url_text)
            except Exception as e:
                st.error(f"Something went wrong: {e}")
                st.stop()

        label = result["label"]
        phishing_prob = result["phishing_probability"]

        st.divider()

        if label == "Phishing":
            st.error(f"### 🚨 Verdict: {label}")
        else:
            st.success(f"### ✅ Verdict: {label}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Phishing probability", f"{phishing_prob:.1%}")
        with col2:
            st.metric("Legitimate probability", f"{result['legitimate_probability']:.1%}")

        st.progress(phishing_prob, text="Phishing likelihood")

        st.divider()
        st.subheader("How the model weighed each signal")
        st.caption(
            "The fusion gate decides how much to trust the email-text branch "
            "vs. the URL branch for this specific prediction."
        )
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.metric(
                "Email text weight",
                f"{result['gate_weight_text']:.0%}",
                help="How much this prediction relied on the email body." if result["has_text"]
                     else "No email text was provided.",
            )
        with gcol2:
            st.metric(
                "URL weight",
                f"{result['gate_weight_url']:.0%}",
                help="How much this prediction relied on the URL." if result["has_url"]
                     else "No URL was provided.",
            )

        if not result["has_text"]:
            st.info("No email text provided -- prediction based on URL alone.")
        if not result["has_url"]:
            st.info("No URL provided -- prediction based on email text alone.")

        if "url_features" in result:
            st.divider()
            st.subheader("URL surface features")
            st.caption("Interpretable signals extracted from the URL, for transparency (not directly used by the model's decision beyond what CANINE learned).")
            feats = result["url_features"]
            fcol1, fcol2, fcol3 = st.columns(3)
            with fcol1:
                st.write("**Host**", feats["host"])
                st.write("**HTTPS**", "✅ Yes" if feats["is_https"] else "❌ No")
            with fcol2:
                st.write("**Length**", feats["length"])
                st.write("**Hyphens in host**", feats["num_hyphens"])
            with fcol3:
                st.write("**IP-based host**", "⚠️ Yes" if feats["has_ip_host"] else "No")
                st.write("**Punycode**", "⚠️ Yes" if feats["is_punycode"] else "No")

st.divider()
st.caption(
    "PhishGuard is a research/educational project and should not be used as a sole "
    "safeguard against phishing. Always verify suspicious emails through official channels."
)