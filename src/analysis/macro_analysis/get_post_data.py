 
import requests
import pandas as pd
import os
from tqdm import tqdm
import time

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
UA = "windows:ada-project:v1.0 (by /u/***REMOVED***)" 
CLIENT_ID = "***REMOVED***"
CLIENT_SECRET = "***REMOVED***"
USERNAME = "***REMOVED***"
PASSWORD = "***REMOVED***"

def get_post_data(path):

    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {"grant_type": "password", "username": USERNAME, "password": PASSWORD, "scope": "read"}
    headers = {"User-Agent": UA}
    r = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    r.raise_for_status()
    token = r.json()["access_token"]

    headers = {"Authorization": f"bearer {token}", "User-Agent": UA}

    df = load_reddit_df(path)
    found_posts = []
    batch_size = 100 
    for i, post_id in enumerate(tqdm(df['POST_ID'])):
        url = f"https://oauth.reddit.com/api/info/?id=t3_{post_id}"
        r = requests.get(url, headers=headers)
        data = r.json()
        children = data.get('data', {}).get('children', [])
        if children:
            found_posts.append([post_id, children[0]['data']['selftext']])
        time.sleep(1)
        
        if (i + 1) % batch_size == 0 and found_posts:
            found_df = pd.DataFrame(found_posts, columns=["POST_ID", "POST_TEXT"])
            found_df.to_csv("found_posts.csv", mode='a', header=not os.path.exists("found_posts.csv"), index=False)
            found_posts = []
        
        if i == 300:
            return
            
    if found_posts:
        found_df = pd.DataFrame(found_posts, columns=["POST_ID", "POST_TEXT"])
        found_df.to_csv("found_posts.csv", mode='a', header=not os.path.exists("found_posts.csv"), index=False)

if __name__ == "__main__":
    get_post_data(os.path.join(DATA_FOLDER, 'reddit_df_body.pkl'))