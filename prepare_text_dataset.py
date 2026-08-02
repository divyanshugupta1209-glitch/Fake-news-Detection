# prepare_text_dataset.py
import pandas as pd
from sklearn.model_selection import train_test_split

# Load raw datasets
gossip_fake = pd.read_csv("data/raw/gossipcop_fake.csv")
gossip_real = pd.read_csv("data/raw/gossipcop_real.csv")
politifact_fake = pd.read_csv("data/raw/politifact_fake.csv")
politifact_real = pd.read_csv("data/raw/politifact_real.csv")

# Add labels
gossip_fake['label'] = 'FAKE'
gossip_real['label'] = 'REAL'
politifact_fake['label'] = 'FAKE'
politifact_real['label'] = 'REAL'

# Ensure 'content' exists (fallback to title if missing)
for df in [gossip_fake, gossip_real, politifact_fake, politifact_real]:
    if 'content' not in df.columns:
        df['content'] = df['title']

# Combine all
all_data = pd.concat([gossip_fake, gossip_real, politifact_fake, politifact_real], ignore_index=True)
all_data = all_data.sample(frac=1, random_state=42)  # Shuffle

# Split into train & test
train_df, test_df = train_test_split(all_data, test_size=0.2, random_state=42, stratify=all_data['label'])

# Save processed CSVs
train_df[['title', 'content', 'label']].to_csv("data/processed/text_train.csv", index=False)
test_df[['title', 'content', 'label']].to_csv("data/processed/text_test.csv", index=False)

print("✅ Train dataset saved to data/processed/text_train.csv")
print("✅ Test dataset saved to data/processed/text_test.csv")
