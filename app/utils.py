import re
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

stop_words = set(stopwords.words("english"))

def preprocess_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = ' '.join([word for word in text.split() if word not in stop_words])
    return text

def fuse_predictions(text_pred, srct_out, rmc_out, scs_out, uga_out, img_pred="UNCERTAIN"):
    votes = [text_pred, srct_out, rmc_out, scs_out, uga_out]
    if img_pred != "UNCERTAIN":
        votes.append(img_pred)

    fake_count = votes.count("FAKE")
    real_count = votes.count("REAL")
    uncertain_count = votes.count("UNCERTAIN")

    if uncertain_count > 2:
        return "UNCERTAIN"
    return "FAKE" if fake_count > real_count else "REAL"
