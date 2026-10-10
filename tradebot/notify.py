"""Phone push notifications via ntfy.sh (free, no account).

Install the ntfy app, subscribe to your topic, and set NTFY_TOPIC to the same name.
Anyone who knows the topic name can read it, so make it hard to guess.
"""
import logging
import os
import urllib.request

log = logging.getLogger("tradebot")


def notify(title: str, message: str, priority: str = "default", tags: str = ""):
    topic = os.getenv("NTFY_TOPIC", "").strip()
    if not topic:
        return
    req = urllib.request.Request(
        f"https://ntfy.sh/{topic}",
        data=message.encode(),
        headers={"Title": title, "Priority": priority, "Tags": tags},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception as e:  # an alert failing must never stop trading
        log.warning("Notification failed: %s", e)
