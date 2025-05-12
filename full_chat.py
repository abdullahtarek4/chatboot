import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses, util
from torch.utils.data import DataLoader
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from deep_translator import GoogleTranslator
import torch
import os
from transformers import BertTokenizer, BertForSequenceClassification
import joblib
from torch.nn.functional import softmax
from sentence_transformers import CrossEncoder


class ComplaintMatcher:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path).dropna(subset=["Reports_clean", "Resolution_clean"])
        self.model = SentenceTransformer("fine_tuned_bert_complaints")
        self.analyzer = SentimentIntensityAnalyzer()
        
        # Initialize Cross-Encoder for similar solutions (lazy loading)
        self.cross_encoder = None

        # Load genre classifier model and assets
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

        print("Encoding all complaints... (one time only)")
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
        """Filter the dataset by predicted genre"""
        if genre not in self.df['Genre'].unique():
            return pd.DataFrame()  # Return empty if genre not found
        return self.df[self.df['Genre'] == genre].copy()

    def find_similar_solutions(self, report, genre, top_n=3):
        """Find most similar solutions with 3-stage sorting:
        1. Filter by genre
        2. Sort by Cross-Encoder similarity
        3. Final sort by satisfaction_score
        """
        # Lazy load Cross-Encoder model
        if self.cross_encoder is None:
            self.cross_encoder = CrossEncoder('cross-encoder/stsb-roberta-base')

        # STAGE 1: Filter by genre first
        genre_df = self.filter_by_genre(genre)
        if genre_df.empty:
            return pd.DataFrame()

        # STAGE 2: Calculate Cross-Encoder similarities
        reports = genre_df['Reports_clean'].tolist()
        pairs = [(report, other_report) for other_report in reports]
        similarities = self.cross_encoder.predict(pairs)
        genre_df['cross_similarity'] = similarities

        # STAGE 3: Sort by similarity THEN satisfaction_score
        genre_df = genre_df.sort_values(
            ['cross_similarity', 'satisfaction_score'],
            ascending=[False, False]
        ).head(top_n)

        return genre_df[['Reports_clean', 'Resolution_clean', 'cross_similarity', 'satisfaction_score']]

    def get_satisfaction_feedback(self, user_input, genre, corrected_input):
        """Handle user satisfaction feedback with enhanced sorting"""
        while True:
            print("\nAre you satisfied with the solution?")
            print("1. Satisfied")
            print("2. Neutral")
            print("3. Dissatisfied")
            
            try:
                choice = int(input("Please enter your choice (1-3): "))
                if choice not in [1, 2, 3]:
                    raise ValueError
            except ValueError:
                print("Invalid input. Please enter 1, 2, or 3.")
                continue
            
            if choice in [1, 2]:
                return "Thank you for your feedback!"
            else:
                # Get solutions sorted by: Genre → Cross-Encoder → Satisfaction Score
                similar_solutions = self.find_similar_solutions(corrected_input, genre, top_n=1)
                
                if similar_solutions.empty:
                    return "We couldn't find alternative solutions. Please try rephrasing your complaint."
                
                print("\nHere are the top alternative solutions (sorted by relevance & satisfaction):")
                for i, (_, row) in enumerate(similar_solutions.iterrows(), 1):
                    print(f"\nOption {i}:")
                    print(f"Similarity: {row['cross_similarity']:.2f}")
                    print(f"Satisfaction Score: {row['satisfaction_score']}")
                    print(f"Original Report: {row['Reports_clean'][:150]}...")
                    print(f"Solution: {row['Resolution_clean']}")
                
                print("\nDid any of these solutions help?")
                print("1. Yes (choose one)")
                print("2. No, none helped")
                
                try:
                    followup_choice = int(input("Please enter your choice (1-2): "))
                    if followup_choice not in [1, 2]:
                        raise ValueError
                except ValueError:
                    print("Invalid input. Returning to main menu.")
                    return "Please try again with a different complaint description."
                
                if followup_choice == 1:
                    return "We're glad one of the alternatives worked for you!"
                else:
                    return "We're sorry we couldn't help. Please contact support for further assistance."

    def get_response(self, user_input, top_n=1, threshold=0.4):
        print("🤖 Hello! How can I help you today?\n")

        # Translate Arabic input to English
        try:
            translated_input = GoogleTranslator(source='auto', target='en').translate(user_input)
            original_lang = "ar" if translated_input != user_input else "en"
        except Exception:
            print("⚠ Proceeding with original input.")
            translated_input = user_input
            original_lang = "en"

        # Genre prediction
        genre, genre_confidence = self.predict_genre(translated_input)

        # Grammar correction
        corrected_input = self.correct_text(translated_input)

        # Encode & analyze sentiment
        input_embedding = self.model.encode(corrected_input, convert_to_tensor=True)
        input_sentiment = self.analyzer.polarity_scores(corrected_input)["compound"]

        # Match against complaints
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
                f"Predicted Genre: {genre} (Confidence: {genre_confidence})\n\n"
                f"No good match found. Try rephrasing.\n"
                f"Input Sentiment: {input_sentiment}\n"
                f"Corrected Input: {corrected_input}"
            )
            final_response = GoogleTranslator(source='en', target='ar').translate(response) if original_lang == "ar" else response
            print(final_response)
            return final_response

        top_matches = sorted(scores, key=lambda x: x["similarity"], reverse=True)[:top_n]

        results = []
        for match in top_matches:
            row = self.df.loc[match["index"]]
            result = (
                f"Predicted Genre: {genre} (Confidence: {genre_confidence})\n\n"
                f"Matched Complaint:\n{row['Reports_clean']}\n\n"
                f"Suggested Resolution:\n{row['Resolution_clean']}\n\n"
                f"Similarity Score: {round(match['similarity'], 3)}\n"
                f"Sentiment Match:\n"
                f"  - User Input: {round(input_sentiment, 2)}\n"
                f"  - Matched Complaint: {round(match['sentiment'], 2)}\n"
                f"Corrected Input:\n{corrected_input}\n"
            )
            results.append(result)
        
        header = f"📂 Predicted Genre: {genre} (Confidence: {genre_confidence})\n\n"
        matched_response = results[0] if top_n == 1 else "\n\n".join(results)
        final_response = header + matched_response

        # Translate back to Arabic if needed
        if original_lang == "ar":
            try:
                # Only translate the matched response part, keep the genre header in English
                translated_response = GoogleTranslator(source='en', target='ar').translate(matched_response)
                final_response = header + translated_response
            except:
                final_response += "\n\n⚠ (Could not translate to Arabic.)"

        # Print the response first
        print(final_response)
        
        # Then ask for satisfaction feedback
        feedback_response = self.get_satisfaction_feedback(user_input, genre, corrected_input)
        print(feedback_response)
        
        return final_response