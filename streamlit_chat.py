import pandas as pd
from sentence_transformers import SentenceTransformer, util, CrossEncoder
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from deep_translator import GoogleTranslator
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from torch.nn.functional import softmax


class ComplaintMatcher:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path).dropna(subset=["Reports_clean", "Resolution_clean"])
        self.model = SentenceTransformer("fine_tuned_bert_complaints")
        self.analyzer = SentimentIntensityAnalyzer()
        self.cross_encoder = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer_genre = BertTokenizer.from_pretrained("bert-base-uncased")
        self.label_encoder = joblib.load("model_assets/label_encoder.joblib")
        self.genre_model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=len(self.label_encoder.classes_)
        ).to(self.device)
        self.genre_model.load_state_dict(torch.load("model_assets/best_model.pt")["model_state_dict"])
        self.genre_model.eval()
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            self.genre_model = torch.compile(self.genre_model)

        self.df["embedding"] = self.df["Reports_clean"].apply(lambda x: self.model.encode(x, convert_to_tensor=True))

    def correct_text(self, text):
        return str(TextBlob(text).correct())

    def predict_genre(self, text):
        with torch.no_grad(), torch.inference_mode():
            inputs = self.tokenizer_genre(
                text,
                truncation=True,
                padding='max_length',
                max_length=128,
                return_tensors="pt"
            ).to(self.device)

            with torch.backends.cuda.sdp_kernel(enable_flash=True):
                outputs = self.genre_model(**inputs)

            probs = softmax(outputs.logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)

            prediction = self.label_encoder.inverse_transform(pred_idx.cpu().numpy())[0]
            confidence = round(confidence.item(), 4)

            return prediction, confidence

    def filter_by_genre(self, genre):
        if genre not in self.df['Genre'].unique():
            return pd.DataFrame()
        return self.df[self.df['Genre'] == genre].copy()

    def find_similar_solutions(self, report, genre, top_n=3):
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

        genre_df = self.filter_by_genre(genre)
        if genre_df.empty:
            return pd.DataFrame()

        reports = genre_df['Reports_clean'].tolist()
        pairs = [(report, other_report) for other_report in reports]
        similarities = self.cross_encoder.predict(pairs)
        genre_df['cross_similarity'] = similarities

        genre_df = genre_df.sort_values(
            ['cross_similarity', 'satisfaction_score'],
            ascending=[False, False]
        ).head(top_n)

        return genre_df[['Reports_clean', 'Resolution_clean', 'cross_similarity', 'satisfaction_score']]

    def get_response(self, user_input, top_n=1, threshold=0.4):
        try:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
            original_lang = "ar" if translated_input != user_input else "en"
        except Exception:
            translated_input = user_input
            original_lang = "en"

        genre, genre_confidence = self.predict_genre(translated_input)
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

        results = []
        if scores:
            top_matches = sorted(scores, key=lambda x: x["similarity"], reverse=True)[:top_n]
            for match in top_matches:
                row = self.df.loc[match["index"]]
                results.append({
                    "matched_complaint": row["Reports_clean"],
                    "resolution": row["Resolution_clean"],
                    "similarity": round(match["similarity"], 3),
                    "user_sentiment": round(input_sentiment, 2),
                    "match_sentiment": round(match["sentiment"], 2),
                    "genre": genre,
                    "confidence": genre_confidence,
                    "corrected_input": corrected_input,
                    "original_lang": original_lang
                })
        return results

    def get_alternative_solutions(self, corrected_input, genre, top_n=1):
        alternatives = self.find_similar_solutions(corrected_input, genre, top_n=top_n)
        return alternatives.to_dict(orient='records') if not alternatives.empty else []
