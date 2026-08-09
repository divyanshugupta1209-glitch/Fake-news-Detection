import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
from PIL import Image
import pandas as pd
from datetime import datetime
import time

from utils.news_api import fetch_articles
from utils import detection, db, pdf_report
from utils.image_utils import compress_image, get_image_info
from utils.cache import get_cached_result, save_to_cache, get_cache_stats

# ── Page config ───────────────────────────────────────────
st.set_page_config(page_title="NEO-REFUTE", page_icon="📰", layout="wide")

# ── Header ────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;padding:20px;background:#111827;
border-radius:12px;margin-bottom:20px'>
    <h1 style='color:#22c55e;margin-bottom:5px;'>📰 NEO-REFUTE</h1>
    <p style='color:#d1d5db;font-size:17px;'>
        Multimodal Fake News Detection System (Text + Image)
    </p>
</div>""", unsafe_allow_html=True)

db.init_db()

# ── Sidebar ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ System Settings")
    uga_threshold = st.slider("Confidence Threshold", 0.45, 0.80, 0.60, 0.01)
    page_size     = st.slider("Articles to fetch", 1, 6, 3)
    
    st.markdown("---")
    
    # Cache stats
    cache_stats = get_cache_stats()
    st.markdown("**📦 Cache Status**")
    st.metric("Cached Results", cache_stats['valid'])
    if cache_stats['expired'] > 0:
        st.caption(f"⏰ {cache_stats['expired']} expired entries")
    
    st.markdown("---")
    st.markdown("""
**NEO-REFUTE Analysis Engines**
- 🧠 NEO-REFUTE AI Engine (Text)
- 🖼️ Visual Consistency Engine
- 📄 Document Scanner (OCR)
- 🤖 Synthetic Media Detector
- 🔍 Deep Language Model
""")
    st.markdown("---")
    
    # Performance tip
    st.info("💡 **Tip:** Results are cached for 7 days. Analyze the same claim twice for instant results!")
    
    st.caption("🚀 NEO-REFUTE v2.5 — Multimodal Intelligence")


# ══════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════

def show_verdict(label, score):
    cm = {"REAL":"#16a34a","FAKE":"#dc2626","UNCERTAIN":"#ea580c"}
    lu = str(label).upper()
    c  = cm.get(lu, "#6b7280")
    vt = {
        "REAL":      "✅  REAL — This claim appears CREDIBLE",
        "FAKE":      "❌  FAKE — This claim appears MISLEADING",
        "UNCERTAIN": "⚠️  UNCERTAIN — Insufficient evidence to verify",
    }.get(lu, lu)
    st.markdown(f"""
<div style='padding:18px;background:{c};color:white;text-align:center;
font-size:22px;border-radius:10px;margin-bottom:6px;font-weight:bold'>
    {vt}
</div>
<div style='padding:8px;background:{c}22;color:{c};text-align:center;
font-size:14px;border-radius:8px;margin-bottom:10px;border:1px solid {c}'>
    Confidence Score: {score*100:.1f}%
</div>""", unsafe_allow_html=True)
    st.progress(min(max(float(score), 0.0), 1.0))


def show_breakdown(text_score, image_score, details):
    st.markdown("### 🔬 Detailed Analysis Breakdown")
    if image_score is not None:
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📝 Text Analysis**")
            st.progress(min(max(float(text_score), 0.0), 1.0))
            st.caption(f"{text_score*100:.2f}%")
        with c2:
            st.markdown("**🖼️ Image Analysis**")
            st.progress(min(max(float(image_score), 0.0), 1.0))
            st.caption(f"{image_score*100:.2f}%")
        with c3:
            fs = details.get("combined_score", 0.5)
            st.markdown("**⚖️ Final Score**")
            st.progress(min(max(float(fs), 0.0), 1.0))
            st.caption(f"{fs*100:.2f}%")
        st.info(
            f"**Fusion:** (0.7 × {text_score*100:.1f}%) + "
            f"(0.3 × {image_score*100:.1f}%) = "
            f"{details.get('combined_score',0.5)*100:.1f}%"
        )
    else:
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**📝 Text Analysis**")
            st.progress(min(max(float(text_score), 0.0), 1.0))
            st.caption(f"{text_score*100:.2f}%")
        with c2:
            st.markdown("**🖼️ Image Analysis**")
            st.info("No image — text-only mode")


def show_explanation(exp: str):
    lines = [l for l in exp.splitlines()
             if not l.lower().strip().startswith(("label:", "score:"))]
    cleaned = "\n".join(lines).strip()
    st.markdown("### 🧠 Analysis & Reasoning")
    st.markdown(f"""
<div style='padding:15px;background:#1f2937;color:#d1d5db;
border-radius:10px;font-size:15px;line-height:1.7'>
    {cleaned.replace(chr(10),"<br>")}
</div>""", unsafe_allow_html=True)


def show_articles(articles):
    st.markdown("### 📰 Related News Articles")
    if not articles:
        st.info("No related news articles found.")
        return
    for a in articles:
        st.markdown(
            f"**{a.get('title','No title')}**  \n"
            f"{a.get('description') or ''}  \n"
            f"[Read more]({a.get('url','#')}) — Source: {a.get('source','Unknown')}"
        )
        st.markdown("---")


def show_no_source(claim):
    st.markdown("### 📰 Related News Articles")
    st.warning(
        f"⚠️ No Verified Sources Found\n\n"
        f"This claim does not appear in any verified news source:\n\n"
        f"**\"{claim}\"**"
    )


# ══════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════
tabs = st.tabs(["🔍 Check News", "📊 Trends", "🏆 Fake Sources"])

# ──────────────────────────────────────────────────────────
# TAB 1: CHECK NEWS
# ──────────────────────────────────────────────────────────
with tabs[0]:
    st.subheader("🧠 Verify a News Claim")
    st.caption("Enter a news headline or claim to analyze its authenticity")

    claim = st.text_area("📝 News Claim or Headline",
                         placeholder="e.g. Scientists confirm coffee cures cancer",
                         height=120)

    uploaded = st.file_uploader("🖼️ Upload Related Image (Optional)",
                                type=["jpg","jpeg","png"])
    image = None
    original_size = None
    compressed_size = None
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        original_size = image.size
        
        # Show original
        st.markdown("**📷 Original Image**")
        st.image(image, use_column_width=True)
        
        # Compress
        with st.spinner("🗜️ Compressing image for faster processing..."):
            image = compress_image(image)
            compressed_size = image.size
        
        # Show compression stats
        orig_pixels = original_size[0] * original_size[1]
        comp_pixels = compressed_size[0] * compressed_size[1]
        reduction = (1 - comp_pixels / orig_pixels) * 100
        
        st.success(
            f"✅ Image compressed: {original_size[0]}×{original_size[1]} → "
            f"{compressed_size[0]}×{compressed_size[1]} "
            f"({reduction:.0f}% reduction)"
        )

    if st.button("🚀 Run NEO-REFUTE", use_container_width=True):
        if not claim.strip():
            st.warning("⚠️ Please enter a claim.")
        else:
            # ── Check cache first ──────────────────────────
            cache_key = claim.strip()
            has_img = image is not None
            cached = get_cached_result(cache_key, has_img)
            
            if cached:
                st.success("⚡ Using cached result (instant!)")
                
                # Extract from cache
                final_label = cached['label']
                final_score = cached['score']
                details     = cached['details']
                t_score     = cached['text_score']
                t_label     = cached.get('text_label', final_label)
                exp_txt     = cached['explanation']
                articles    = cached.get('articles', [])
                url         = cached.get('url', '')
                
            else:
                # Progress indicators
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    # ── 1. Text analysis (SLOWEST) ────────────
                    status_text.text("🧠 Step 1/5: Analyzing text (15-20s)...")
                    progress_bar.progress(10)
                    
                    hf      = detection.query_hf_model(claim)
                    t_label = str(hf.get("label","UNCERTAIN")).upper()
                    t_score = float(hf.get("score", 0.5))
                    exp_txt = hf.get("explanation","No explanation.")
                    
                    progress_bar.progress(40)

                    # ── 2. Articles ───────────────────────────
                    status_text.text("📰 Step 2/5: Fetching news articles (2-3s)...")
                    articles = fetch_articles(claim, page_size) \
                               if t_label == "REAL" else []
                    
                    progress_bar.progress(55)

                    # ── 3. Context ────────────────────────────
                    if articles:
                        a   = articles[0]
                        txt = (a.get("title") or "") + " " + (a.get("description") or "")
                        url = a.get("url","")
                        src = a.get("source","")
                    else:
                        txt, url, src = claim, "", ""
                    
                    progress_bar.progress(60)

                    # ── 4. Fusion ─────────────────────────────
                    if image is not None:
                        status_text.text("🖼️ Step 3/5: Analyzing image (5-8s)...")
                    else:
                        status_text.text("⚖️ Step 3/5: Computing final score...")
                    
                    final_label, final_score, details = detection.fuse_predictions(
                        claim_text=claim, article_text=txt,
                        url=url, image=image,
                        hf_result=hf, uga_threshold=uga_threshold
                    )
                    
                    progress_bar.progress(85)

                    # 5. Keep the fused verdict even when no image is uploaded
                    if image is None:
                        details["image_score"] = None

                    # Always inject text score
                    details["hf_score"] = t_score
                    
                    progress_bar.progress(90)
                    status_text.text("💾 Step 4/5: Saving results to cache...")

                    # ── 6. Save to cache ──────────────────────
                    cache_result = {
                        'label': final_label,
                        'score': final_score,
                        'details': details,
                        'text_score': t_score,
                        'text_label': t_label,
                        'explanation': exp_txt,
                        'articles': articles,
                        'url': url,
                    }
                    save_to_cache(cache_key, cache_result, has_img)

                    # ── 7. Save to DB ─────────────────────────
                    status_text.text("🗄️ Step 5/5: Saving to database...")
                    db.save_detection(claim, txt, url, src,
                                      final_label, final_score, t_score)
                    
                    progress_bar.progress(100)
                    status_text.text("✅ Analysis complete!")
                    time.sleep(0.5)
                    
                finally:
                    # Clean up progress indicators
                    progress_bar.empty()
                    status_text.empty()

            # ══════════════════════════════════════════
            # DISPLAY RESULTS
            # ══════════════════════════════════════════

            # Main verdict
            show_verdict(final_label, final_score)

            # Score breakdown
            image_score = details.get("image_score")
            show_breakdown(t_score, image_score, details)

            # OCR Results
            if details.get("ocr_result") and details["ocr_result"].get("has_text"):
                ocr = details["ocr_result"]
                st.markdown("### 📄 Document Scanner — Text Found in Image")
                if ocr.get("warning"): st.error(ocr["warning"])
                if ocr.get("info"):    st.info(ocr["info"])
                if ocr.get("extracted_text"):
                    with st.expander("📝 View Extracted Text"):
                        st.text_area("", ocr["extracted_text"],
                                     height=150, disabled=True,
                                     key="ocr_txt")

            # AI Detection
            ai = details.get("ai_detection_result")
            if ai:
                st.markdown("### 🤖 Synthetic Media Detection")
                if ai.get("warning"): st.error(ai["warning"])
                if ai.get("is_ai"):   st.warning(ai.get("info",""))
                else:                 st.success(ai.get("info",""))
                col1, col2 = st.columns(2)
                with col1: st.metric("Synthetic Probability", f"{ai.get('confidence',0):.0%}")
                with col2: st.metric("Authenticity Score",    f"{ai.get('score',0):.2f}")

            # Explanation
            show_explanation(exp_txt)

            # Articles
            if t_label == "REAL": 
                show_articles(articles)
            else:
                show_no_source(claim)

            # PDF
            st.markdown("---")
            st.markdown("### 📄 Export Analysis Report")
            try:
                pdf = pdf_report.generate_pdf_report(
                    claim_text=claim,
                    final_label=final_label,
                    final_score=final_score,
                    details=details,
                    image=image
                )
                st.download_button(
                    "📥 Download Full Analysis Report (PDF)",
                    data=pdf,
                    file_name=f"neo_refute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                    mime="application/pdf",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Report generation failed: {e}")

# ──────────────────────────────────────────────────────────
# TAB 2: TRENDS
# ──────────────────────────────────────────────────────────
with tabs[1]:
    st.subheader("📊 Detection History")
    df = db.recent_detections(200)
    if df.empty: st.info("No history yet.")
    else: st.dataframe(df, use_container_width=True, hide_index=True)

# ──────────────────────────────────────────────────────────
# TAB 3: FAKE SOURCES
# ──────────────────────────────────────────────────────────
with tabs[2]:
    st.subheader("🏆 Most Frequently Detected Fake Sources")
    top = db.top_fake_sources(20)
    if top.empty: st.info("No fake sources yet.")
    else: st.bar_chart(top.set_index("source")["cnt"], use_container_width=True)
