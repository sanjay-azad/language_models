import json, datetime, pytz

now = datetime.datetime.now(pytz.timezone('Asia/Kolkata')).isoformat()

def log(user_message, assistant_message):
    log_entry = json.dumps({
        "time": now,
        "user": user_message,
        "bot": assistant_message
    })
    with open('chats.log', 'a') as f:
        f.write(log_entry + '\n')