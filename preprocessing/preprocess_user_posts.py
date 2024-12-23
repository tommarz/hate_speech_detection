import pandas as pd
import gzip
import simplejson
import re
from io import BytesIO
from tqdm import tqdm
import zipfile
from collections import defaultdict
from datetime import datetime
import pickle
import yaml


def convert_to_timestamp(date_str):
    """
    Converts a Twitter date string to an integer timestamp.

    Args:
        date_str (str): The date string (e.g., "Tue Oct 10 20:19:24 +0000 2023").

    Returns:
        int: The Unix timestamp as an integer.
    """
    try:
        # Parse the Twitter date format
        dt = datetime.strptime(date_str, "%a %b %d %H:%M:%S %z %Y")
        # Convert to Unix timestamp and return as integer
        return int(dt.timestamp())
    except ValueError as e:
        print(f"Error parsing date: {date_str} - {e}")
        return None


def preprocess_echo_posts(data_path, ignore_punct=True, ignore_rt=False):
    """
    Processes tweets from a nested ZIP file containing gzipped JSON files.

    Args:
        data_path (str): Path to the ZIP file containing tweets.
        ignore_punct (bool): Whether to remove punctuation from the tweets.
        ignore_rt (bool): Whether to ignore retweets.

    Returns:
        pd.DataFrame: DataFrame containing interaction data for replies and retweets.
        dict: Dictionary containing sorted tweets per user as triplets (timestamp, tweet_id, text).
    """
    tweets_data = []
    all_users = defaultdict(list)

    # Regular expressions for cleaning
    URL_RE = re.compile(r"https?://\S+|www\.\S+")
    PUNCT_RE = re.compile(r"[^\w\s]")

    with zipfile.ZipFile(data_path, "r") as zfile:
        for name in tqdm(zfile.namelist()):
            if re.search(r"\.gz$", name):
                zfiledata = BytesIO(zfile.read(name))
                with gzip.open(zfiledata) as zf:
                    for line in zf:
                        json_tweet = simplejson.loads(line)
                        user_id_str = lookup(json_tweet, "user.id_str")
                        tweet_id = lookup(json_tweet, "id_str")
                        timestamp = convert_to_timestamp(
                            lookup(json_tweet, "created_at")
                        )

                        if not user_id_str or not tweet_id or not timestamp:
                            continue

                        # Extract text
                        if "retweeted_status" in json_tweet.keys():  # Retweet
                            if (
                                lookup(json_tweet, "retweeted_status.truncated")
                                is False
                            ):
                                text = lookup(json_tweet, "retweeted_status.full_text")
                            else:
                                text = lookup(
                                    json_tweet,
                                    "retweeted_status.extended_tweet.full_text",
                                )
                        elif lookup(json_tweet, "truncated") is False:
                            text = lookup(json_tweet, "full_text")
                        else:
                            text = lookup(json_tweet, "extended_tweet.full_text")

                        # Clean text
                        text = URL_RE.sub("", text)
                        if ignore_punct:
                            text = PUNCT_RE.sub(" ", text)

                        # Add tweet to user's list
                        all_users[user_id_str].append((timestamp, tweet_id, text))

                        # Skip retweets if specified
                        if ignore_rt and "retweeted_status" in json_tweet.keys():
                            continue

                        # Prioritize interaction types
                        interaction_types = []

                        # Check if it's a reply
                        in_reply_to_user_id = lookup(
                            json_tweet, "in_reply_to_user_id_str"
                        )
                        if in_reply_to_user_id and user_id_str != in_reply_to_user_id:
                            interaction_types.append(
                                {
                                    "type": "reply",
                                    "source": user_id_str,
                                    "target": in_reply_to_user_id,
                                    "text": text,
                                    "timestamp": timestamp,
                                    "tweet_id": tweet_id,
                                }
                            )

                        # Check if it's a retweet
                        if "retweeted_status" in json_tweet.keys():
                            retweeted_user_id = lookup(
                                json_tweet, "retweeted_status.user.id_str"
                            )
                            if user_id_str != retweeted_user_id:
                                interaction_types.append(
                                    {
                                        "type": "retweet",
                                        "source": user_id_str,
                                        "target": retweeted_user_id,
                                        "text": text,
                                        "timestamp": timestamp,
                                        "tweet_id": tweet_id,
                                    }
                                )

                        # Check if it's a mention
                        mentions = lookup(json_tweet, "entities.user_mentions")
                        if mentions:
                            for mention in mentions:
                                mentioned_user_id = lookup(mention, "id_str")
                                if user_id_str != mentioned_user_id:
                                    interaction_types.append(
                                        {
                                            "type": "mention",
                                            "source": user_id_str,
                                            "target": mentioned_user_id,
                                            "text": text,
                                            "timestamp": timestamp,
                                            "tweet_id": tweet_id,
                                        }
                                    )

                        # Deduplicate interactions based on precedence
                        if "reply" in [x["type"] for x in interaction_types]:
                            tweets_data.append(
                                next(
                                    x for x in interaction_types if x["type"] == "reply"
                                )
                            )
                        elif "retweet" in [x["type"] for x in interaction_types]:
                            tweets_data.append(
                                next(
                                    x
                                    for x in interaction_types
                                    if x["type"] == "retweet"
                                )
                            )
                        elif "mention" in [x["type"] for x in interaction_types]:
                            tweets_data.append(
                                next(
                                    x
                                    for x in interaction_types
                                    if x["type"] == "mention"
                                )
                            )

    # Create DataFrame for interactions
    interactions_df = pd.DataFrame(tweets_data)

    # Sort all users' tweets by timestamp
    tweets_per_user = {
        user: sorted(tweets, key=lambda x: x[0]) for user, tweets in all_users.items()
    }

    return interactions_df, tweets_per_user


# Utility function to safely fetch nested dictionary keys
def lookup(dct, path, default=None):
    keys = path.split(".")
    for key in keys:
        dct = dct.get(key, default)
        if dct is default:
            break
    return dct


import os
import pickle
import re
import json
from tqdm import tqdm
from collections import defaultdict
import pandas as pd

def preprocess_gab_posts(data_path, users_info_path, ignore_punct=False, ignore_rt=False):
    """
    Preprocesses the Gab dataset and returns:
    1. A dictionary of posts per user sorted by timestamp/post ID, where each element is a tuple of (timestamp, ID, text).
    2. A DataFrame of interactions with source and target user IDs, interaction type, and text (if available).

    Args:
        base_path (str): Base path to the dataset.
        data_path (str): Path to the data file.
        ignore_punct (bool): Whether to ignore punctuation in text.
        ignore_rt (bool): Whether to ignore retweets.

    Returns:
        dict: Dictionary of posts per user.
        pd.DataFrame: DataFrame of interactions.
    """
    # Load username to ID mapping

    user_info_df = pd.read_pickle(users_info_path)
    username2id_mapping = user_info_df['id'].to_dict()

    # Regular expressions for URLs, mentions, and punctuation
    URL_RE = re.compile(r'https?://\S+')
    MENTION_RE = re.compile(r'@\w+')
    PUNCT_RE = re.compile(r'[!"#$%&\'()*+,\-./:;<=>?@[\\]^_`{|}~]')

    # Data structures
    posts_per_user = defaultdict(list)
    interactions = []

    # Process the data file
    with open(data_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin):
            json_content = json.loads(line.strip())
            user_id_str = str(json_content["actuser"]["id"])
            json_type = json_content["type"]
            timestamp = json_content["post"]["created_at"]  # Assuming "created_at" contains the timestamp
            post_id = json_content["post"]["id"]  # Assuming "id" is the post ID
            text = json_content['post']['body']

            # Remove URLs
            text = URL_RE.sub('', text)

            # Remove punctuation if required
            if ignore_punct:
                text = PUNCT_RE.sub(' ', text)

            if json_type == 'repost' and not ignore_rt:
                # Handle reposts (like RTs)
                user_reposted_id = str(json_content['post']['user']['id'])
                if user_id_str != user_reposted_id:
                    interactions.append({
                        "source": user_id_str,
                        "target": user_reposted_id,
                        "type": "repost",
                        "text": text
                    })
            elif json_type == 'post':
                # Handle mentions in posts
                mentions = [mention.replace('@', '') for mention in MENTION_RE.findall(text)]
                for mention in mentions:
                    user_mentioned_id = username2id_mapping.get(mention, mention)  # Map username to ID if possible
                    if user_mentioned_id != user_id_str:
                        interactions.append({
                            "source": user_id_str,
                            "target": user_mentioned_id,
                            "type": "mention",
                            "text": text
                        })

            # Add post to the user's list of posts
            posts_per_user[user_id_str].append((timestamp, post_id, text))

    # Sort posts per user by timestamp/post ID
    for user_id in posts_per_user:
        posts_per_user[user_id].sort(key=lambda x: (x[0], x[1]))

    # Convert interactions to a DataFrame
    interactions_df = pd.DataFrame(interactions)

    return interactions_df, posts_per_user


def preprocess_parler_posts(data_path, comments_path, echos_path, in_reply_to_dict):
    # Load data
    df = pd.read_csv(data_path, sep="\t", encoding="utf-8")

    # Remove duplicates of the same post text and same person ID
    df.drop_duplicates(subset=["text", "user_id"], keep="first", inplace=True)

    # Remove URLs from the tweet's text
    df["text"] = df["text"].str.replace(
        r"(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]*)([\/\w\.-]*)*\/\?\S", "", regex=True
    )

    # Create a dictionary to store all interactions
    interaction_data = []
    user_posts = defaultdict(list)

    # Add user tweets to user_posts only
    for index, row in df.iterrows():
        tweet_text = row["text"]
        user_posts[row["user_id"]].append({"text_id": index, "text": tweet_text})

    # Sort posts per user by text_id
    for user in user_posts:
        user_posts[user] = sorted(user_posts[user], key=lambda x: x["text_id"])

    # Load interaction dictionaries
    with open(comments_path, "rb") as f:
        mentions_dict = pickle.load(f)

    with open(echos_path, "rb") as f:
        retweets_dict = pickle.load(f)

    # Handle in_reply_to
    for user_id, mentioned_users in in_reply_to_dict.items():
        for mentioned_user_id, texts in mentioned_users.items():
            for text in texts:
                interaction_data.append(
                    {
                        "source": user_id,
                        "target": mentioned_user_id,
                        "interaction_type": "reply",
                        "text": text,
                    }
                )

    # Process retweets
    for user_id, mentioned_users in retweets_dict.items():
        for mentioned_user_id, texts in mentioned_users.items():
            for text in texts if "parler" in data_path else range(len(texts)):
                interaction_data.append(
                    {
                        "source": user_id,
                        "target": mentioned_user_id,
                        "interaction_type": "retweet",
                        "text": text if isinstance(text, str) else None,
                    }
                )

    # Process mentions
    for user_id, mentioned_users in mentions_dict.items():
        for mentioned_user_id, texts in mentioned_users.items():
            for text in texts if "parler" in data_path else range(len(texts)):
                interaction_data.append(
                    {
                        "source": user_id,
                        "target": mentioned_user_id,
                        "interaction_type": "mention",
                        "text": text if isinstance(text, str) else None,
                    }
                )

    # Create a dataframe from the interaction data
    interactions_df = pd.DataFrame(interaction_data)

    return interactions_df, user_posts


def process_tweets(data_path, output_path):
    # Load data
    df = pd.read_csv(data_path)

    # Preprocess tweets
    df["text"] = df["text"].str.lower()
    df["text"] = df["text"].str.replace(r"http\S+", "", regex=True)
    df["text"] = df["text"].str.replace(r"[^a-zA-Z\s]", "", regex=True)

    # Group tweets by username
    grouped = df.groupby("user_id")["text"].apply(lambda x: " ".join(x)).reset_index()

    # Save to output file
    grouped.to_csv(output_path, index=False)

    return grouped


def preprocess_posts(dataset_name, **kwargs):

    # Strategy pattern for dataset preprocessing
    strategy = {"parler": preprocess_parler_posts, "echo": preprocess_echo_posts, "gab": preprocess_gab_posts}

    if dataset_name not in strategy:
        raise ValueError(f"No preprocessing function for dataset {dataset_name}.")

    return strategy[dataset_name](**kwargs)


# Example usage
# result = process_dataset('config.yaml', 'parler_posts')
