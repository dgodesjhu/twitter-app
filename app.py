import streamlit as st
import openai
import faiss
import pandas as pd
import numpy as np
import os

# Load API key from Hugging Face secrets
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai.api_key = OPENAI_API_KEY

st.write("🔍 Debug: API Key Exists:", bool(api_key))  # Should print True if the key is found
st.write("🔍 Debug: API Key (first 5 chars):", api_key[:5] if api_key else "Not Found")

# Load dataset of highly liked tweets
@st.cache_data
def load_tweets():
    df = pd.read_csv("highly_liked_tweets.csv")
    return df

# Convert text to embeddings (Updated API Call)
def get_embedding(text):
    response = openai.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    )
    return np.array(response.data[0].embedding)

# Build FAISS index for similarity search
def build_faiss_index(df):
    embeddings = np.array([get_embedding(tweet) for tweet in df["tweet"]])
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

# Find most similar tweet in the database
def find_similar_tweet(index, df, tweet):
    query_embedding = get_embedding(tweet)
    D, I = index.search(np.array([query_embedding]), k=1)  # Find closest match
    best_match = df.iloc[I[0][0]]["tweet"]
    similarity_score = 1 - (D[0][0] / 2)  # Normalize cosine similarity
    return best_match, similarity_score

# Generate AI tweet suggestions (Updated API Call)
def generate_tweets(topic):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert in crafting engaging tweets."},
            {"role": "user", "content": f"Generate 3 highly engaging tweets about {topic}. Each should be under 280 characters and designed to maximize engagement."}
        ],
        max_tokens=200,
        temperature=0.8
    )
    return [choice.message.content.strip() for choice in response.choices]

# Suggest refinements for a tweet (Updated API Call)
def refine_tweet(tweet, similar_tweet):
    response = openai.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are an expert social media strategist."},
            {"role": "user", "content": f"Refine this tweet to make it more engaging based on this high-performing tweet: {similar_tweet}. Ensure it's concise and under 280 characters."},
            {"role": "assistant", "content": tweet}
        ],
        max_tokens=100,
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Streamlit UI
st.title("🚀 AI-Powered Tweet Optimizer")

# Step 1: Select Input Mode
input_mode = st.radio("Choose how to start:", ("Enter My Own Tweets", "Generate Tweets with AI"))

if input_mode == "Enter My Own Tweets":
    # User inputs their own tweets
    user_tweets = []
    for i in range(1, 4):
        tweet = st.text_area(f"Enter Tweet {i}:", "")
        user_tweets.append(tweet)

    # Store tweets in session
    if st.button("Analyze My Tweets"):
        st.session_state["tweets"] = [tweet for tweet in user_tweets if tweet.strip()]

elif input_mode == "Generate Tweets with AI":
    # AI-Generated Tweets
    topic = st.text_input("Enter a topic for AI-generated tweets:", "AI in marketing")

    if st.button("Generate Tweets"):
        st.session_state["tweets"] = generate_tweets(topic)

# Step 2: Analyze Tweets
if "tweets" in st.session_state and st.session_state["tweets"]:
    df = load_tweets()
    index, _ = build_faiss_index(df)

    st.subheader("Your Tweets (Manual or AI-Generated)")
    
    for i, tweet in enumerate(st.session_state["tweets"]):
        st.write(f"🔹 **Tweet {i+1}:** {tweet}")  # Properly format tweet number

        # Find the closest high-engagement tweet
        similar_tweet, similarity_score = find_similar_tweet(index, df, tweet)
        st.write(f"🔍 **Closest High-Engagement Tweet:** {similar_tweet}")
        st.write(f"🧠 **Similarity Score:** {round(similarity_score, 2)}")

        # Suggest refinement
        refined_tweet = refine_tweet(tweet, similar_tweet)
        st.write(f"💡 **AI-Suggested Refinement:** {refined_tweet}")

        st.divider()

st.write("📌 Adjust your prompt and refine your tweets to improve engagement!")
