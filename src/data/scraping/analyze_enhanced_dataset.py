import requests
import pandas as pd
import os
from tqdm import tqdm
import time
import sys
sys.path.append(os.path.join(os.getcwd(), '..', '..', '..'))

from src.utils.data_utils import load_reddit_df

DATA_FOLDER = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data')
from reddit_config import UA, CLIENT_ID, CLIENT_SECRET, USERNAME, PASSWORD

def sanitize(s: str):
    """
    Sanitize string by removing problematic characters for CSV storage.
    
    Args:
        s: Input string to sanitize
    
    Returns:
        str or None: Sanitized string with quotes, commas, newlines, and special characters removed, or None if input is not a string
    """
    if not isinstance(s, str):
        return None
    s = s.replace('"', "'")
    s = s.replace(',', ' ')
    s = s.replace('\n', ' ')
    s = s.replace('\r', ' ')
    s = s.replace('\t', ' ')
    s = s.replace(';', '')
    s = s.replace("'", "")
    return s 

def compare_post_ids(enhanced_df, old_df, token):
    """
    Compare post IDs between enhanced and old dataframes, filling in missing posts from Reddit API.
    
    Args:
        enhanced_df: Enhanced dataframe with additional Reddit API data
        old_df: Original dataframe to compare against
        token: Reddit API access token
    
    Returns:
        pandas.DataFrame: Enhanced dataframe with missing posts filled in from API
    """    
    defining_columns = ['SOURCE_SUBREDDIT', 'TARGET_SUBREDDIT', 'POST_ID', 'LINK_SENTIMENT']
    enhanced_sub = enhanced_df[defining_columns]
    old_sub = old_df[defining_columns]

    merged = enhanced_sub.merge(old_sub, how="outer", on=defining_columns, indicator=True)

    only_in_df1 = merged.query("_merge == 'left_only'")
    only_in_df2 = merged.query("_merge == 'right_only'")

    print(len(only_in_df1), len(only_in_df2))
    new_posts = []

    for _, i in tqdm(only_in_df2.iterrows()):
        if i["POST_ID"] in enhanced_df["POST_ID"].values:
            print("skip")
        else:
            if not enhanced_df.loc[
            enhanced_df["permalink"].fillna("").str.contains(str(i.POST_ID), na=False, regex=False),
            ["POST_ID", "permalink"]].empty:
               print("Filling post ID")
               
               permalink_id = enhanced_df.loc[enhanced_df["permalink"].fillna("").str.contains(str(i.POST_ID), na=False, regex=False),"permalink"].iloc[0].split("/")[4]
               mask = enhanced_df["permalink"].fillna("").str.contains(str(i.POST_ID), na=False, regex=False)
               enhanced_df.loc[mask, "POST_ID"] = permalink_id
            else:
                print("Looking for post")
                new_df_dict = check_missing_post(i["POST_ID"], token)
                old_df_dict = old_df.loc[(old_df["POST_ID"] == i["POST_ID"]) &
                                         (old_df["SOURCE_SUBREDDIT"] == i["SOURCE_SUBREDDIT"]) &
                                         (old_df["TARGET_SUBREDDIT"] == i["TARGET_SUBREDDIT"])].iloc[0].to_dict()
                old_df_dict.update(new_df_dict)
                new_posts.append(old_df_dict)

    found_df = pd.DataFrame(new_posts)
    enhanced_df = pd.concat([enhanced_df, found_df], ignore_index=True)


    return enhanced_df

def check_missing_post(post_id, token):
    """
    Fetch post data from Reddit API for a missing post ID.
    
    Args:
        post_id: Reddit post ID to fetch
        token: Reddit API access token
    
    Returns:
        dict: Dictionary containing post data fields, empty dict if post not found
    """
    headers = {"Authorization": f"bearer {token}", "User-Agent": UA}
    url = f"https://oauth.reddit.com/api/info/?id=t3_{post_id}"
    attempt = 0
    while True:
        try:
            r = requests.get(url, headers=headers)
            r.raise_for_status()
            break
        except Exception as e:
            if attempt == 200:
                break
            attempt += 1
            wait = 3
            print(f"Request failed for {post_id} (attempt {attempt}): {e}. Retrying in {wait}s...")
            time.sleep(wait)
    data = r.json()
    new_df_dict = {}
    children = data.get('data', {}).get('children', [])
    if children:
        d = children[0]['data']
        new_df_dict = {
            "POST_ID": str(post_id),
            "approved_at_utc": d.get("approved_at_utc"),
            "subreddit": d.get("subreddit"),
            "user_reports": d.get("user_reports"),
            "saved": d.get("saved"),
            "mod_reason_title": sanitize(d.get("mod_reason_title")),
            "gilded": d.get("gilded"),
            "clicked": d.get("clicked"),
            "link_flair_richtext": d.get("link_flair_richtext"),
            "subreddit_name_prefixed": d.get("subreddit_name_prefixed"),
            "hidden": d.get("hidden"),
            "pwls": d.get("pwls"),
            "link_flair_css_class": sanitize(d.get("link_flair_css_class")),
            "downs": d.get("downs"),
            #   "thumbnail_height": d.get("thumbnail_height"),
            "top_awarded_type": d.get("top_awarded_type"),
            "hide_score": d.get("hide_score"),
            "name": sanitize(d.get("name")),
            "quarantine": d.get("quarantine"),
            "link_flair_text_color": sanitize(d.get("link_flair_text_color")),
            "upvote_ratio": d.get("upvote_ratio"),
            "author_flair_background_color": sanitize(d.get("author_flair_background_color")),
            "subreddit_type": d.get("subreddit_type"),
            "ups": d.get("ups"),
            "total_awards_received": d.get("total_awards_received"),
                # "media_embed": d.get("media_embed"),
                # "thumbnail_width": d.get("thumbnail_width"),
            "author_flair_template_id": sanitize(d.get("author_flair_template_id")),
            "is_original_content": d.get("is_original_content"),
                # "secure_media": d.get("secure_media"),
            "is_reddit_media_domain": d.get("is_reddit_media_domain"),
            "is_meta": d.get("is_meta"),
            "category": d.get("category"),
                # "secure_media_embed": d.get("secure_media_embed"),
            "link_flair_text": sanitize(d.get("link_flair_text")),
            "can_mod_post": d.get("can_mod_post"),
            "score": d.get("score"),
            "approved_by": sanitize(d.get("approved_by")),
            "is_created_from_ads_ui": d.get("is_created_from_ads_ui"),
                # "thumbnail": d.get("thumbnail"),
            "edited": d.get("edited"),
            "author_flair_css_class": d.get("author_flair_css_class"),
            "gildings": d.get("gildings"),
            "content_categories": d.get("content_categories"),
            "is_self": d.get("is_self"),
            "mod_note": d.get("mod_note"),
            "created": sanitize(d.get("created")),
            "link_flair_type": sanitize(d.get("link_flair_type")),
            "wls": d.get("wls"),
            "removed_by_category": d.get("removed_by_category"),
            "banned_by": d.get("banned_by"),
            "domain": d.get("domain"),
            "allow_live_comments": d.get("allow_live_comments"),
            #   "selftext_html": d.get("selftext_html"),
            "likes": d.get("likes"),
            "suggested_sort": d.get("suggested_sort"),
            "banned_at_utc": d.get("banned_at_utc"),
            "view_count": d.get("view_count"),
            "archived": d.get("archived"),
            "no_follow": d.get("no_follow"),
            "is_crosspostable": d.get("is_crosspostable"),
            "pinned": d.get("pinned"),
            "over_18": d.get("over_18"),
            "all_awardings": d.get("all_awardings"),
            "awarders": d.get("awarders"),
            "media_only": d.get("media_only"),
            "can_gild": d.get("can_gild"),
            "spoiler": d.get("spoiler"),
            "locked": d.get("locked"),
            "author_flair_text": sanitize(d.get("author_flair_text")),
            "treatment_tags": sanitize(d.get("treatment_tags")),
            "visited": d.get("visited"),
            "removed_by": d.get("removed_by"),
            "num_reports": d.get("num_reports"),
            "distinguished": d.get("distinguished"),
            "subreddit_id": d.get("subreddit_id"),
            "author_is_blocked": d.get("author_is_blocked"),
            "mod_reason_by": d.get("mod_reason_by"),
            "removal_reason": d.get("removal_reason"),
                #"link_flair_background_color": d.get("link_flair_background_color"),
            "id": d.get("id"),
            "is_robot_indexable": d.get("is_robot_indexable"),
            "num_duplicates": d.get("num_duplicates"),
            "report_reasons": d.get("report_reasons"),
            "author": sanitize(d.get("author")),
            "discussion_type": d.get("discussion_type"),
            "num_comments": d.get("num_comments"),
            "send_replies": d.get("send_replies"),
                # "media": d.get("media"),
            "contest_mode": d.get("contest_mode"),
            "author_flair_text_color": d.get("author_flair_text_color"),
            "permalink": sanitize(d.get("permalink")),
            "stickied": d.get("stickied"),
            "url": sanitize(d.get("url")),
            "subreddit_subscribers": d.get("subreddit_subscribers"),
            "created_utc": d.get("created_utc"),
            "num_crossposts": d.get("num_crossposts"),
            "mod_reports": d.get("mod_reports"),
            "is_video": d.get("is_video"),
            "title": sanitize(d.get("title")),
            "selftext": sanitize(d.get("selftext")),
        }
    else:
        print(f"No data found for POST_ID: {post_id}")
    
    return new_df_dict

def get_token():
    """
    Get Reddit API access token using OAuth credentials.
    
    Returns:
        str: Reddit API access token
    """
    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {"grant_type": "password", "username": USERNAME, "password": PASSWORD, "scope": "read"}
    headers = {"User-Agent": UA}
    r = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    r.raise_for_status()
    return r.json()["access_token"]

def compare_rows(old_df, enhanced_df):
    """
    Compare rows between old and enhanced dataframes to detect inconsistencies.
    
    Args:
        old_df: Original dataframe
        enhanced_df: Enhanced dataframe to compare
    """
    cnt = 0
    for i in tqdm(range(min(len(old_df), len(enhanced_df)))):
        old_row = old_df.iloc[i]
        new_row = enhanced_df.iloc[i]
        if old_row["POST_ID"] != new_row["POST_ID"] or old_row["SOURCE_SUBREDDIT"] != new_row["SOURCE_SUBREDDIT"]:
            print("Inconsistency detected in row", str(i))
            print(old_row, new_row)
            cnt += 1
            if cnt == 10:
                break
        
def char_key(s: str):
    """
    Generate a sortable key for natural sorting of strings with mixed characters and numbers.
    
    Args:
        s: Input string to generate key for
    
    Returns:
        tuple: Tuple of tuples, where each inner tuple is (type, value) - type 0 for digits, 1 for letters, 2 for other
    """
    s = str(s)
    key = []
    for ch in s:
        if ch.isdigit():
            key.append((0, int(ch))) 
        elif ch.isalpha():
            key.append((1, ch.lower()))
        else:
            key.append((2, ch)) 
    return tuple(key)


if __name__ == "__main__":
    enhanced_df = load_reddit_df(os.path.join(DATA_FOLDER, "enhanced_reddit_df.pkl"))
    old_df = load_reddit_df(os.path.join(DATA_FOLDER, "reddit_df_merged.pkl"))

    # Setting the POST_ID value to string because we had some issues with scientific formatting
    enhanced_df["POST_ID"] = enhanced_df["POST_ID"].astype(str)
    old_df["POST_ID"] = old_df["POST_ID"].astype(str)
    token = get_token()
    enhanced_df = compare_post_ids(enhanced_df, old_df, token)

    # Manually changing some POST_IDs because Excel interpreted them in scientific notation
    enhanced_df.loc[enhanced_df["POST_ID"] == "2.31E+10", "POST_ID"] = "2313e7"
    enhanced_df.loc[enhanced_df["POST_ID"] == "2.00E+236", "POST_ID"] = "2e0236"
    old_df.loc[old_df["POST_ID"] == "2.31E+10", "POST_ID"] = "2313e7"
    old_df.loc[old_df["POST_ID"] == "2.00E+236", "POST_ID"] = "2e0236"

    # Dropping rows that are duplicates - checking these four columns is enough as there cant be two lines with identical values here
    enhanced_df = enhanced_df.drop_duplicates(subset=["SOURCE_SUBREDDIT", "TARGET_SUBREDDIT", "POST_ID", "LINK_SENTIMENT"])
    body_df = enhanced_df[enhanced_df["source_type"]=="body"]
    title_df = enhanced_df[enhanced_df["source_type"]=="title"]
    
    # Sorting body and title as the original df was sorted (first by post id then target subreddit)
    # Merging them after because that is the how we merged the original tsvs
    title_df = title_df.sort_values(
        by=["POST_ID", "TARGET_SUBREDDIT"],
        key=lambda col: (
            col.astype(str).map(char_key)
            if col.name == "POST_ID"
            else col.str.lower()
        )
    )
    body_df = body_df.sort_values(
        by=["POST_ID", "TARGET_SUBREDDIT"],
        key=lambda col: (
            col.astype(str).map(char_key)
            if col.name == "POST_ID"
            else col.str.lower()
        )
    )
    enhanced_df = pd.concat([body_df, title_df], ignore_index=True)

    # Comparing the first few elements to ensure same contents / sorting as in the original file
    # compare_rows(old_df, enhanced_df)
    # Creating the final pkl and csv files
    enhanced_df.to_pickle(os.path.join(DATA_FOLDER, "enhanced_reddit_df_finalized.pkl"))
    enhanced_df.to_csv(os.path.join(DATA_FOLDER, "enhanced_reddit_df_finalized.csv"))