 
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

def get_post_data(path, start_index=0, batch_size=100):

    auth = requests.auth.HTTPBasicAuth(CLIENT_ID, CLIENT_SECRET)
    data = {"grant_type": "password", "username": USERNAME, "password": PASSWORD, "scope": "read"}
    headers = {"User-Agent": UA}
    r = requests.post("https://www.reddit.com/api/v1/access_token", auth=auth, data=data, headers=headers)
    r.raise_for_status()
    token = r.json()["access_token"]

    headers = {"Authorization": f"bearer {token}", "User-Agent": UA}

    df = load_reddit_df(path)
    df = df.iloc[start_index:].reset_index(drop=True)
    found_posts = []
    for i, post_id in enumerate(tqdm(df['POST_ID'])):
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
        children = data.get('data', {}).get('children', [])
        if children:
            d = children[0]['data']
            old_df_dict = df.iloc[i].to_dict()
            new_df_dict = {
                "POST_ID": post_id,
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

        time.sleep(0.4)
        old_df_dict.update(new_df_dict)
        
        found_posts.append(old_df_dict)
        
        if (i + 1) % batch_size == 0 and found_posts:
            found_df = pd.DataFrame(found_posts)
            found_df.to_csv("data/enhanced.csv", mode='a', header=not os.path.exists("data/enhanced.csv"), index=False)
            found_posts = []
        
        if i == 301:
            return
            
    if found_posts:
        found_df = pd.DataFrame(found_posts)
        found_df.to_csv("data/enhanced.csv", mode='a', header=not os.path.exists("data/enhanced.csv"), index=False)

if __name__ == "__main__":
    get_post_data(os.path.join(DATA_FOLDER, 'reddit_df_merged.pkl'))