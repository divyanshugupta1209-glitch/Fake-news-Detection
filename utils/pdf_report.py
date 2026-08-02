# utils/pdf_report.py

import io
import re
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer,
    Table, TableStyle, HRFlowable
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.graphics.shapes import Drawing, Rect, String

print("✅ PDF Report module loaded")

# ── Colors ────────────────────────────────────────────────
C_GREEN      = colors.HexColor("#16a34a")
C_GREEN_D    = colors.HexColor("#14532d")
C_RED        = colors.HexColor("#dc2626")
C_RED_D      = colors.HexColor("#991b1b")
C_ORANGE     = colors.HexColor("#ea580c")
C_ORANGE_D   = colors.HexColor("#c2410c")
C_DARK       = colors.HexColor("#1e293b")
C_DARK2      = colors.HexColor("#334155")
C_GRAY       = colors.HexColor("#64748b")
C_LGRAY      = colors.HexColor("#cbd5e1")
C_LIGHT      = colors.HexColor("#f8fafc")
C_LIGHT2     = colors.HexColor("#f1f5f9")
C_WHITE      = colors.white
C_BLUE       = colors.HexColor("#2563eb")
C_LBLUE      = colors.HexColor("#eff6ff")
C_YELLOW     = colors.HexColor("#d97706")


def vc(label):
    l = str(label).upper()
    return C_GREEN if l == "REAL" else C_RED if l == "FAKE" else C_ORANGE

def vd(label):
    l = str(label).upper()
    return C_GREEN_D if l == "REAL" else C_RED_D if l == "FAKE" else C_ORANGE_D

def vbg(label):
    l = str(label).upper()
    if l == "REAL": return colors.HexColor("#f0fdf4")
    if l == "FAKE": return colors.HexColor("#fef2f2")
    return colors.HexColor("#fff7ed")

def pct(v):
    try:    return f"{float(v)*100:.1f}%"
    except: return "—"

def fs(v, d=0.5):
    """Safe float score with fallback"""
    try:
        val = float(v)
        return max(0.0, min(1.0, val)) if val != 0.0 else d
    except:
        return d


# ── Score bar ─────────────────────────────────────────────
def bar(score, w=240, h=14, bc=None):
    score = fs(score)
    if bc is None:
        bc = C_GREEN if score >= 0.6 else C_ORANGE if score >= 0.4 else C_RED
    d = Drawing(w, h)
    d.add(Rect(0, 0, w, h,
               fillColor=colors.HexColor("#e2e8f0"),
               strokeColor=C_LGRAY, strokeWidth=0.5))
    fill = int(w * score)
    if fill > 2:
        d.add(Rect(1, 1, fill - 1, h - 2,
                   fillColor=bc, strokeColor=None))
    d.add(String(w/2, h/2 - 3.5, f"{score*100:.1f}%",
                 fontSize=7.5, fontName="Helvetica-Bold",
                 fillColor=C_WHITE if score > 0.35 else C_DARK,
                 textAnchor="middle"))
    return d


# ── Verdict box ───────────────────────────────────────────
def verdict_box(label, score, page_w=17.4):
    W   = page_w * cm
    H   = 75
    mid = W * 0.54

    label_u = str(label).upper()
    vcolor  = vc(label_u)
    vdark   = vd(label_u)

    desc = {
        "REAL":      "This claim appears to be CREDIBLE",
        "FAKE":      "This claim appears to be MISLEADING",
        "UNCERTAIN": "Insufficient evidence to verify",
    }.get(label_u, "")

    d = Drawing(W, H)

    d.add(Rect(0, 0, mid, H, fillColor=vcolor, strokeColor=None))
    d.add(Rect(mid, 0, W - mid, H, fillColor=vdark, strokeColor=None))

    d.add(String(mid / 2, H/2 + 10, label_u,
                 fontSize=32, fontName="Helvetica-Bold",
                 fillColor=C_WHITE, textAnchor="middle"))
    
    desc_color = colors.HexColor("#fecaca") if label_u == "FAKE" \
            else colors.HexColor("#bbf7d0") if label_u == "REAL" \
            else colors.HexColor("#fed7aa")
    
    d.add(String(mid / 2, H/2 - 16, desc,
                 fontSize=8.5, fontName="Helvetica",
                 fillColor=desc_color, textAnchor="middle"))

    rx = mid + (W - mid) / 2
    d.add(String(rx, H/2 + 12, "Confidence Score",
                 fontSize=9, fontName="Helvetica",
                 fillColor=desc_color, textAnchor="middle"))
    d.add(String(rx, H/2 - 18, pct(score),
                 fontSize=28, fontName="Helvetica-Bold",
                 fillColor=C_WHITE, textAnchor="middle"))

    return d


# ── Clean explanation ─────────────────────────────────────
def clean_exp(raw):
    if not raw:
        return "No explanation available."
    lines, out = raw.splitlines(), []
    for line in lines:
        s = line.strip()
        l = s.lower()
        if not s:
            if out: out.append("")
            continue
        if re.match(r'^label\s*:\s*(real|fake|uncertain)\s*$', l): continue
        if re.match(r'^score\s*:\s*[\d.]+\s*$', l):               continue
        if re.match(r'^[\d.]+\s*$', s):                            continue
        if l.startswith("explanation:"):
            after = s[len("explanation:"):].strip()
            if after: out.append(after)
            continue
        out.append(s)
    r = "\n".join(out).strip()
    return r if r else "No explanation available."


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════
def generate_pdf_report(claim_text, final_label, final_score,
                        details, image=None):

    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4,
                            rightMargin=1.8*cm, leftMargin=1.8*cm,
                            topMargin=1.5*cm,   bottomMargin=1.5*cm)
    styles = getSampleStyleSheet()
    story  = []

    label_u = str(final_label).upper()
    fsc     = fs(final_score, d=0.5)

    # ── Pull scores ───────────────────────────────────────
    txs = fs(details.get("hf_score") or
             details.get("text_score") or
             fsc, d=fsc)

    img_score  = details.get("image_score")
    ocr_res    = details.get("ocr_result")
    ai_res     = details.get("ai_detection_result")
    raw_expln  = details.get("hf_explanation", "") or ""

    # ── Styles ────────────────────────────────────────────
    def S(n, **kw):
        return ParagraphStyle(n, parent=styles["Normal"], **kw)

    hdr_t  = S("ht", fontSize=22, textColor=C_WHITE,
               fontName="Helvetica-Bold", alignment=TA_CENTER)
    hdr_s  = S("hs", fontSize=8.5,  textColor=colors.HexColor("#94a3b8"),
               fontName="Helvetica", alignment=TA_CENTER, leading=12)
    meta_s = S("ms", fontSize=8.5,textColor=C_DARK2,
               fontName="Helvetica", leading=13)
    sec_s  = S("sc", fontSize=11, textColor=C_DARK,
               fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
    bod_s  = S("bd", fontSize=9,  textColor=C_DARK2,
               fontName="Helvetica", leading=14)
    bld_s  = S("bl", fontSize=9,  textColor=C_DARK,
               fontName="Helvetica-Bold", leading=14)
    sm_s   = S("sm", fontSize=8,  textColor=C_GRAY,
               fontName="Helvetica", leading=12)
    mo_s   = S("mo", fontSize=8.5,textColor=C_DARK2,
               fontName="Courier",   leading=13, leftIndent=6)
    rsn_s  = S("rs", fontSize=9,  textColor=C_DARK2,
               fontName="Helvetica", leading=15)
    ft_s   = S("ft", fontSize=7.5,textColor=C_GRAY,
               fontName="Helvetica")
    ft_r   = S("fr", fontSize=7.5,textColor=C_GRAY,
               fontName="Helvetica", alignment=TA_RIGHT)
    clm_s  = S("cl", fontSize=10, textColor=C_DARK,
               fontName="Helvetica-Bold", leading=15)

    W = 17.4

    def hr(color=C_LGRAY, thick=1.0):
        return HRFlowable(width="100%", thickness=thick,
                          color=color, spaceAfter=5)

    def sp(h=0.3):
        return Spacer(1, h*cm)

    # ══════════════════════════════════════
    # HEADER (FIXED: more spacing to prevent overlap)
    # ══════════════════════════════════════
    hdr = Table(
        [[Paragraph("NEO-REFUTE", hdr_t)],
         [Paragraph("Multimodal Fake News Detection System  —  Analysis Report", hdr_s)]],
        colWidths=[W*cm]
    )
    hdr.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_DARK),
        ("TOPPADDING",    (0,0),(0,0),   16),
        ("BOTTOMPADDING", (0,0),(0,0),   8),
        ("TOPPADDING",    (0,1),(0,1),   6),
        ("BOTTOMPADDING", (0,1),(0,1),   14),
        ("LEFTPADDING",   (0,0),(-1,-1), 16),
        ("RIGHTPADDING",  (0,0),(-1,-1), 16),
    ]))
    story += [hdr, sp(0.3)]

    # ══════════════════════════════════════
    # META
    # ══════════════════════════════════════
    meta = Table([[
        Paragraph(f"<b>Report Date:</b>  {datetime.now().strftime('%d %B %Y   %H:%M')}", meta_s),
        Paragraph("<b>System:</b>  NEO-REFUTE v2.5  |  Multimodal Intelligence", meta_s),
    ]], colWidths=[8.7*cm, 8.7*cm])
    meta.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_LIGHT2),
        ("GRID",          (0,0),(-1,-1), 0.5, C_LGRAY),
        ("TOPPADDING",    (0,0),(-1,-1), 6),
        ("BOTTOMPADDING", (0,0),(-1,-1), 6),
        ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ("RIGHTPADDING",  (0,0),(-1,-1), 10),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story += [meta, sp(0.4)]

    # ══════════════════════════════════════
    # ANALYZED CLAIM
    # ══════════════════════════════════════
    story += [Paragraph("ANALYZED CLAIM", sec_s), hr(C_BLUE, 1.5)]
    disp = (claim_text[:297]+"...") if len(claim_text)>300 else claim_text
    ctbl = Table([[Paragraph(f'"{disp}"', clm_s)]], colWidths=[W*cm])
    ctbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), C_LBLUE),
        ("LINEBEFORE",    (0,0),(0,-1),  3, C_BLUE),
        ("LINEAFTER",     (0,0),(0,-1),  0.5, C_LGRAY),
        ("LINEABOVE",     (0,0),(-1,0),  0.5, C_LGRAY),
        ("LINEBELOW",     (0,-1),(-1,-1),0.5, C_LGRAY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("RIGHTPADDING",  (0,0),(-1,-1), 14),
    ]))
    story += [ctbl, sp(0.4)]

    # ══════════════════════════════════════
    # FINAL VERDICT
    # ══════════════════════════════════════
    story += [Paragraph("FINAL VERDICT", sec_s), hr(vc(label_u), 1.5)]
    story += [verdict_box(label_u, fsc, W), sp(0.4)]

    # ══════════════════════════════════════
    # SCORE BREAKDOWN
    # ══════════════════════════════════════
    story += [Paragraph("SCORE BREAKDOWN", sec_s), hr(C_GRAY, 1.5)]

    rows = [[
        Paragraph("Analysis Engine", bld_s),
        Paragraph("Score",           bld_s),
        Paragraph("Confidence",      bld_s),
        Paragraph("Weight",          bld_s),
    ]]

    def addrow(name, score, weight, indent=False, bc=None):
        n = ("     " + name) if indent else name
        st = sm_s if indent else bod_s
        rows.append([
            Paragraph(n,         st),
            Paragraph(pct(score),st),
            bar(score, bc=bc),
            Paragraph(weight,    st),
        ])

    addrow("Text Analysis", txs, "70%")
    if img_score is not None:
        addrow("Image Analysis (Combined)", fs(img_score), "30%")
    if ocr_res and ocr_res.get("has_text"):
        addrow("Document Scanner (OCR)",
               fs(ocr_res.get("score", 0.7)),
               "9% total", indent=True, bc=C_BLUE)
    if ai_res:
        addrow("Synthetic Media Detector",
               fs(ai_res.get("score", 0.7)),
               "6% total", indent=True, bc=C_YELLOW)

    rows.append([
        Paragraph("FINAL SCORE",              bld_s),
        Paragraph(f"<b>{pct(fsc)}</b>",      bod_s),
        bar(fsc),
        Paragraph("<b>100%</b>",              bod_s),
    ])

    stbl = Table(rows, colWidths=[5.0*cm, 1.5*cm, 6.2*cm, 4.7*cm])
    stbl.setStyle(TableStyle([
        ("BACKGROUND",     (0,0), (-1,0),  C_DARK),
        ("TEXTCOLOR",      (0,0), (-1,0),  C_WHITE),
        ("FONTNAME",       (0,0), (-1,0),  "Helvetica-Bold"),
        ("FONTSIZE",       (0,0), (-1,0),  9),
        ("BACKGROUND",     (0,-1),(-1,-1), C_LIGHT2),
        ("LINEABOVE",      (0,-1),(-1,-1), 1, C_DARK),
        ("TOPPADDING",     (0,0), (-1,-1), 7),
        ("BOTTOMPADDING",  (0,0), (-1,-1), 7),
        ("LEFTPADDING",    (0,0), (-1,-1), 5),
        ("RIGHTPADDING",   (0,0), (-1,-1), 5),
        ("GRID",           (0,0), (-1,-1), 0.5, C_LGRAY),
        ("ROWBACKGROUNDS", (0,1), (-1,-2), [C_WHITE, C_LIGHT]),
        ("VALIGN",         (0,0), (-1,-1), "MIDDLE"),
        ("ALIGN",          (1,0), (1,-1),  "CENTER"),
        ("ALIGN",          (3,0), (3,-1),  "CENTER"),
    ]))
    story += [stbl, sp(0.4)]

    # ══════════════════════════════════════
    # FUSION FORMULA
    # ══════════════════════════════════════
    if img_score is not None:
        imgs = fs(img_score)
        story += [Paragraph("FUSION FORMULA", sec_s), hr(C_GRAY, 1.5)]
        formula = (
            f"Final Score  =  ( 0.70  x  Text Score )  +  ( 0.30  x  Image Score )\n"
            f"             =  ( 0.70  x  {txs:.3f} )  +  ( 0.30  x  {imgs:.3f} )\n"
            f"             =  {0.7*txs:.3f}  +  {0.3*imgs:.3f}\n"
            f"             =  {fsc:.3f}   -->   {label_u}"
        )
        ftbl = Table([[Paragraph(formula, mo_s)]], colWidths=[W*cm])
        ftbl.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C_LIGHT2),
            ("LINEBEFORE",    (0,0),(0,-1),  3, C_DARK2),
            ("LINEABOVE",     (0,0),(-1,0),  0.5, C_LGRAY),
            ("LINEBELOW",     (0,-1),(-1,-1),0.5, C_LGRAY),
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
            ("RIGHTPADDING",  (0,0),(-1,-1), 14),
        ]))
        story += [ftbl, sp(0.4)]

    # ══════════════════════════════════════
    # ANALYSIS & REASONING
    # ══════════════════════════════════════
    explanation = clean_exp(raw_expln)
    story += [Paragraph("ANALYSIS & REASONING", sec_s), hr(C_GRAY, 1.5)]
    etbl = Table(
        [[Paragraph(explanation.replace("\n", "<br/>"), rsn_s)]],
        colWidths=[W*cm]
    )
    etbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0),(-1,-1), vbg(label_u)),
        ("LINEBEFORE",    (0,0),(0,-1),  3, vc(label_u)),
        ("LINEAFTER",     (0,0),(0,-1),  0.5, C_LGRAY),
        ("LINEABOVE",     (0,0),(-1,0),  0.5, C_LGRAY),
        ("LINEBELOW",     (0,-1),(-1,-1),0.5, C_LGRAY),
        ("TOPPADDING",    (0,0),(-1,-1), 10),
        ("BOTTOMPADDING", (0,0),(-1,-1), 10),
        ("LEFTPADDING",   (0,0),(-1,-1), 14),
        ("RIGHTPADDING",  (0,0),(-1,-1), 14),
    ]))
    story += [etbl, sp(0.4)]

    # ══════════════════════════════════════
    # IMAGE ANALYSIS DETAILS
    # ══════════════════════════════════════
    if img_score is not None and (ocr_res or ai_res):
        story += [Paragraph("IMAGE ANALYSIS DETAILS", sec_s), hr(C_GRAY, 1.5)]
        irows = [[
            Paragraph("Engine",  bld_s),
            Paragraph("Result",  bld_s),
            Paragraph("Status",  bld_s),
        ]]
        if ocr_res:
            os2    = fs(ocr_res.get("score", 0.7))
            ostatus= "MATCH" if os2>0.6 else "PARTIAL" if os2>0.3 else "MISMATCH"
            otxt   = "No text in image" if not ocr_res.get("has_text") \
                     else f"{ocr_res.get('match_ratio',0):.0%} keyword overlap"
            irows.append([Paragraph("Document Scanner (OCR)", bod_s),
                          Paragraph(otxt, bod_s),
                          Paragraph(ostatus, bod_s)])
        if ai_res:
            ac     = fs(ai_res.get("confidence", 0))
            astatus= "SYNTHETIC" if ai_res.get("is_ai") else \
                     "UNCERTAIN" if ac>0.4 else "AUTHENTIC"
            irows.append([Paragraph("Synthetic Media Detector", bod_s),
                          Paragraph(f"Probability: {ac:.1%}", bod_s),
                          Paragraph(astatus, bod_s)])
        itbl = Table(irows, colWidths=[5.5*cm, 8*cm, 3.9*cm])
        itbl.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  C_DARK),
            ("TEXTCOLOR",      (0,0),(-1,0),  C_WHITE),
            ("FONTNAME",       (0,0),(-1,0),  "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,0),  9),
            ("TOPPADDING",     (0,0),(-1,-1), 7),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 7),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
            ("RIGHTPADDING",   (0,0),(-1,-1), 8),
            ("GRID",           (0,0),(-1,-1), 0.5, C_LGRAY),
            ("ROWBACKGROUNDS", (0,1),(-1,-1), [C_WHITE, C_LIGHT]),
            ("VALIGN",         (0,0),(-1,-1), "MIDDLE"),
        ]))
        story += [itbl, sp(0.4)]

    # ══════════════════════════════════════
    # FOOTER
    # ══════════════════════════════════════
    story.append(hr(C_LGRAY, 0.5))
    ftbl2 = Table([[
        Paragraph("Generated by NEO-REFUTE v2.5  —  Multimodal Fake News Detection Intelligence", ft_s),
        Paragraph(datetime.now().strftime("%Y-%m-%d   %H:%M"), ft_r),
    ]], colWidths=[10.5*cm, 6.9*cm])
    ftbl2.setStyle(TableStyle([
        ("TOPPADDING",    (0,0),(-1,-1), 2),
        ("BOTTOMPADDING", (0,0),(-1,-1), 2),
        ("LEFTPADDING",   (0,0),(-1,-1), 0),
        ("RIGHTPADDING",  (0,0),(-1,-1), 0),
        ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
    ]))
    story.append(ftbl2)

    doc.build(story)
    pdf = buf.getvalue()
    buf.close()
    return pdf