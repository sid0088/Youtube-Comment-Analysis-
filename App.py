from flask import Flask, render_template, request
import os
import re
import nltk
import pandas as pd
from googleapiclient.discovery import build
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Initialize the Flask app
app = Flask(__name__)

# Set up YouTube API
api_key = "AIzaSyAPbNwqfWHzHpHYCZdZ4NkFjBetG89jtJ4" # Replace with your API key
youtube = build('youtube', 'v3', developerKey=api_key)

# NLTK Data
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

# Function to extract video ID from YouTube URL
def extract_video_id(url):
    if "v=" in url:
        return url.split("v=")[1].split("&")[0]
    elif "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0]
    else:
        raise ValueError("Invalid YouTube URL format")

# Function to get comments from a video
def get_comments(video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet",
        videoId=video_id,
        maxResults=100,
        textFormat="plainText"
    )
    response = request.execute()

    while request is not None:
        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
            comments.append(comment)
        if 'nextPageToken' in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                pageToken=response['nextPageToken'],
                maxResults=100,
                textFormat="plainText"
            )
            response = request.execute()  # Fetch next page
        else:
            break

    return comments

# Function to get video statistics (views, likes, dislikes)
def get_video_stats(video_id):
    request = youtube.videos().list(
        part="statistics",
        id=video_id
    )
    response = request.execute()

    views = response['items'][0]['statistics']['viewCount']
    likes = response['items'][0]['statistics'].get('likeCount', 0)
    dislikes = response['items'][0]['statistics'].get('dislikeCount', 0)

    return views, likes, dislikes

# Sentiment Analysis function
def perform_sentiment_analysis(comments):
    sentiments = SentimentIntensityAnalyzer()
    data = {'Comment': comments}
    df = pd.DataFrame(data)
    
    # Get sentiment scores for each comment
    df["Positive"] = [sentiments.polarity_scores(comment)["pos"] for comment in df["Comment"]]
    df["Negative"] = [sentiments.polarity_scores(comment)["neg"] for comment in df["Comment"]]
    df["Neutral"] = [sentiments.polarity_scores(comment)["neu"] for comment in df["Comment"]]
    df['Compound'] = [sentiments.polarity_scores(comment)["compound"] for comment in df["Comment"]]
    
    # Classify each comment as Positive, Negative, or Neutral
    df["Sentiment"] = df['Compound'].apply(lambda score: 'Positive' if score >= 0.05 else ('Negative' if score <= -0.05 else 'Neutral'))

    # Analyze sentiment counts
    sentiment_counts = df["Sentiment"].value_counts()

    return df, sentiment_counts

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video_url = request.form["video_url"]
        video_id = extract_video_id(video_url)
        
        # Get comments and video statistics
        comments = get_comments(video_id)
        views, likes, dislikes = get_video_stats(video_id)
        
        # Perform sentiment analysis
        result, sentiment_counts = perform_sentiment_analysis(comments)
        
        # Render the result on the webpage
        return render_template("index.html", 
                               comments=comments, 
                               result=result.to_html(classes='table table-striped'), 
                               sentiment_counts=sentiment_counts, 
                               views=views, 
                               likes=likes, 
                               dislikes=dislikes)

    return render_template("index.html", comments=None, result=None, sentiment_counts=None, views=None, likes=None, dislikes=None)

if __name__ == "__main__":
    # Change port number here
    app.run(debug=True, host="0.0.0.0", port=8080)
