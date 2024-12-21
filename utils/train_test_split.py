import pandas as pd
import os
from sklearn.model_selection import train_test_split
import json

main_dir = os.path.join(os.environ['HOME'], 'hate_speech_detection_thesis')
os.chdir(main_dir)
from config import data_config

def split_posts_by_users(labeled_posts_df, labeled_users_series, test_size=0.2, random_state=42):
    # Split the users into train and test sets
    train_users, test_users = train_test_split(
        labeled_users_series.index, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labeled_users_series.values
    )
    
    labeled_posts_with_labeled_users_df = labeled_posts_df.query('username in @labeled_users_series.index')
    # Split the posts dataframe based on the train and test users
    labeled_train_df = labeled_posts_with_labeled_users_df.query('`username` in @train_users')
    labeled_test_df = labeled_posts_with_labeled_users_df.query('`username` in @test_users')
    
    labeled_posts_with_unlabeled_users_df = labeled_posts_df.query('username not in @labeled_users_series.index')
    # Split the posts dataframe directly based on the posts' labels
    train_df_posts, test_df_posts = train_test_split(
        labeled_posts_with_unlabeled_users_df, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labeled_posts_with_unlabeled_users_df['label']
    )
    
    # Concatenate the results
    train_df = pd.concat([labeled_train_df, train_df_posts]).drop_duplicates().reset_index(drop=True)
    test_df = pd.concat([labeled_test_df, test_df_posts]).drop_duplicates().reset_index(drop=True)
    
    return train_df, test_df