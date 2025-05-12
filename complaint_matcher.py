import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from deep_translator import GoogleTranslator
import torch
import os

class ComplaintMatcher:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path).dropna(subset=["Reports_clean", "Resolution_clean"])
        self.model = SentenceTransformer("fine_tuned_bert_complaints")
        self.analyzer = SentimentIntensityAnalyzer()

        print("Encoding all complaints... (one time only)")
        self.df["embedding"] = self.df["Reports_clean"].apply(lambda x: self.model.encode(x, convert_to_tensor=True))
    
    def correct_text(self, text):
        return str(TextBlob(text).correct())

    def get_response(self, user_input, top_n=1, threshold=0.4):
        print("🤖 Hello! How can I help you today?\n")

        # Translate Arabic input to English
        try:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
            original_lang = "ar" if translated_input != user_input else "en"
        except Exception as e:
            print("⚠️ Proceeding with original input.")
            translated_input = user_input
            original_lang = "en"

        corrected_input = self.correct_text(translated_input)
        input_embedding = self.model.encode(corrected_input, convert_to_tensor=True)
        input_sentiment = self.analyzer.polarity_scores(corrected_input)["compound"]

        scores = []
        for idx, row in self.df.iterrows():
            sim = util.cos_sim(input_embedding, row["embedding"]).item()
            if sim >= threshold:
                scores.append({
                    "index": idx,
                    "similarity": sim,
                    "sentiment": self.analyzer.polarity_scores(row["Reports_clean"])["compound"]
                })

        if not scores:
            response = (
                f"No good match found. Try rephrasing.\n"
                f"Input Sentiment: {input_sentiment}\n"
                f"Corrected Input: {corrected_input}"
            )
            return GoogleTranslator(source='en', target='ar').translate(response) if original_lang == "ar" else response

        top_matches = sorted(scores, key=lambda x: x["similarity"], reverse=True)[:top_n]

        results = []
        for match in top_matches:
            row = self.df.loc[match["index"]]
            result = (
                f"Matched Complaint:\n{row['Reports_clean']}\n\n"
                f"Suggested Resolution:\n{row['Resolution_clean']}\n\n"
                f"Similarity Score: {round(match['similarity'], 3)}\n"
                f"Sentiment Match:\n"
                f"  - User Input: {round(input_sentiment, 2)}\n"
                f"  - Matched Complaint: {round(match['sentiment'], 2)}\n"
                f"Corrected Input:\n{corrected_input}\n"
            )
            results.append(result)

        final_response = results[0] if top_n == 1 else "\n\n".join(results)

        # Translate response to Arabic if needed
        if original_lang == "ar":
            try:
                final_response = GoogleTranslator(source='en', target='ar').translate(final_response)
            except:
                final_response += "\n\n⚠️ (Could not translate to Arabic.)"

        return final_response



if __name__ == "__main__":
    df = pd.read_csv("Final_student_tickting_cleaned.csv").dropna(subset=["Reports_clean", "Resolution_clean"])
    
    train_examples = []
    for idx, row in df.iterrows():
        train_examples.append(InputExample(texts=[row["Reports_clean"], row["Resolution_clean"]], label=1.0))
        if idx + 1 < len(df):
            train_examples.append(InputExample(texts=[row["Reports_clean"], df.iloc[idx + 1]["Resolution_clean"]], label=0.0))

    model = SentenceTransformer('all-MiniLM-L6-v2')  # Efficient model
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
    train_loss = losses.CosineSimilarityLoss(model)

    print("Training model...")
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=2,
        warmup_steps=10,
        show_progress_bar=True
    )

    print("Saving model to disk...")
    model.save("fine_tuned_bert_complaints")
