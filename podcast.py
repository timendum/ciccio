import re
import shutil
from collections import namedtuple
from datetime import datetime, timedelta
from os import path, getenv
from urllib.parse import urljoin

import feedendum
import requests
from bs4 import BeautifulSoup

Puntata = namedtuple("Puntata", ["url", "title", "mp3"])

BASE_URL = getenv("BASE_URL", ".")


def find_mp3() -> Puntata | None:
    r = requests.get("https://www.deejay.it/programmi/il-terzo-incomodo/puntate/")
    r.raise_for_status()
    hpuntate = BeautifulSoup(r.text, features="lxml")
    puntate = [urljoin(r.url, x["href"]) for x in hpuntate.css.select("h1 a")]
    puntata = puntate[0]
    r = requests.get(puntata)
    hpuntata = BeautifulSoup(r.text, features="lxml")
    r.raise_for_status()
    mp3r = re.search(r"https:\/\/[-a-zA-Z0-9()_\+\.~\/]*\.mp3", r.text)
    if not mp3r:
        return None
    title = hpuntata.title.text
    if title.count("|") == 2:
        title = title.split("|")[1].strip()
    return Puntata(r.url, title, mp3r[0])


def already_done(p: Puntata, outdir: str) -> bool:
    try:
        feed = feedendum.from_rss_file(path.join(outdir, "terzo-incomodo.xml"))
        return feed.items and p.title in feed.items[0].title
    except OSError:
        return False


def make_feed(p: Puntata, files: list[str], outdir: str) -> None:
    now = datetime.now()
    feed = feedendum.feed.Feed(
        title="Il terzo incomodo - podcast non ufficiale",
        url="https://www.deejay.it/programmi/il-terzo-incomodo/puntate/",
        description="Puntate del terzo incomodo, ma con solo gli spezzoni di parlato.",
        update=now + timedelta(minutes=30),
        _data={
            "image": {
                "url": "https://www.omnycontent.com/d/playlist/60311b15-274a-4e3f-8ba9-ac3000834f37/1636480f-f515-4800-ad4b-b07500779d1f/72cf72a2-49cb-42ef-a619-b07500779d52/image.jpg?t=1694095271&size=Large"
            }
        },
    )
    for i, f in enumerate(files):
        feed.items.append(
            feedendum.feed.FeedItem(
                title=f"{p.title} - {i + 1:0>2}",
                url=f"{p.url}#{i:0>2}",
                id=f"{p.url}#{i:0>2}",
                content=f"Spezzone {i + 1} di {(len(files) + 1)}\nDa {p.mp3}",
                update=now + timedelta(minutes=i),
                _data={
                    "enclosure": {
                        "@url": urljoin(BASE_URL, f),
                        "@type": "audio/mpeg",
                        "@length": str(path.getsize(f)),
                    }
                },
            )
        )
        shutil.move(f, path.join(outdir, f))
    with open(path.join(outdir, "terzo-incomodo.xml"), "w", encoding="utf8") as text_file:
        text_file.write(feedendum.to_rss_string(feed))
    return feed
