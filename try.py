from full_chat import ComplaintMatcher
import warnings
warnings.filterwarnings("ignore", message="Some weights of BertForSequenceClassification")
warnings.filterwarnings("ignore", category=FutureWarning)
bot = ComplaintMatcher("Final_student_sentiment_cleaned.csv")
print(bot.get_response("I feel there should be better Arabic language courses."))
