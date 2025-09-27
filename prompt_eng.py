from transformers import pipeline
# Load a local model (e.g., sentiment-analysis)
sentiment_model = pipeline("sentiment-analysis")
text_generation_model = pipeline("text-generation", model="gpt2")
# 🔍 Task 1: Summarization Prompting

paragraph = """The quick brown fox jumps over the lazy dog to reach the other side of the hill."""

# Zero-shot prompt
zero_shot_prompt = f"Summarize the following paragraph:\n{paragraph}"
print("🔹 Zero-shot Summary:")
print(text_generation_model(zero_shot_prompt, max_length=50)[0]['generated_text'])
# summarization prompting

paragraph = """The quick brown fox jumps over the lazy dog to reach the other side of the hill."""

# Zero-shot prompt
zero_shot_prompt = f"Summarize the following paragraph:\n{paragraph}"
print("🔹 Zero-shot Summary:")
print(text_generation_model(zero_shot_prompt, max_length=50)[0]['generated_text'])

# Few-shot prompt
few_shot_prompt = """Example: 'She went to the store, bought groceries, and came back home in the afternoon.'
Summary: She bought groceries and returned home.

Now summarize the following paragraph:
""" + paragraph
print("\n🔹 Few-shot Summary:")
print(text_generation_model(few_shot_prompt, max_length=50)[0]['generated_text'])
# 🧪 Task 2: Sentiment Classification

review_text = "The new update is fantastic – it made the app so much smoother and I absolutely love it!"

# Basic prompt
print("\n🔹 Basic Sentiment Classification:")
print(sentiment_model(review_text))

# Role-based prompt (simulated)
role_based_prompt = f"You are a sentiment analysis expert. Classify the sentiment of this review: '{review_text}'"
print("\n🔹 Role-based Sentiment Classification (simulated):")
print(text_generation_model(role_based_prompt, max_length=50)[0]['generated_text'])

# 🧠 Task 3: Q&A with Chain-of-Thought

question = "If a bat and a ball cost $1.10 total, and the bat costs $1.00 more than the ball, how much does the ball cost?"

# Direct prompt
direct_prompt = f"Q: {question}\nA:"
print("\n🔹 Direct Answer:")
print(text_generation_model(direct_prompt, max_length=50)[0]['generated_text'])

# Chain-of-thought prompt
cot_prompt = f"Q: {question}\nLet's think step by step."
print("\n🔹 Chain-of-Thought Answer:")
print(text_generation_model(cot_prompt, max_length=150)[0]['generated_text'])
