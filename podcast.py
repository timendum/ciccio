import re
import shutil
from collections import namedtuple
from datetime import datetime, timedelta
from os import getenv, path, unlink
from urllib.parse import urljoin

import feedendum
import requests
from bs4 import BeautifulSoup
from netlify import NetlifyClient

Puntata = namedtuple("Puntata", ["url", "title", "mp3"])

BASE_URL = getenv("BASE_URL", ".")
NETLIFY_TOKEN = getenv("NETLIFY", "")


def find_mp3(puntata) -> Puntata | None:
    if not puntata:
        r = requests.get("https://www.deejay.it/programmi/chiacchiericcio/puntate/")
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
        feed = feedendum.from_rss_file(path.join(outdir, "chiacchiericcio.xml"))
        return feed.items and p.title in feed.items[0].title
    except OSError:
        return False


def make_feed(p: Puntata, files: list[str], outdir: str) -> None:
    now = datetime.now()
    feed = feedendum.feed.Feed(
        title="Chiacchiericcio - podcast non ufficiale",
        url="https://www.deejay.it/programmi/chiacchiericcio/",
        description="Puntate di Chiacchiericcio, ma con solo gli spezzoni di parlato.",
        update=now + timedelta(minutes=30),
        _data={
            "image": {
                "url": "https://cdn.gelestatic.it/deejay/sites/2/2023/09/IlTerzoIncomodo__Cover-1200x627-640x334.jpg"
            }
        },
    )
    for i, f in enumerate(files):
        feed.items.append(
            feedendum.feed.FeedItem(
                title=f"{p.title} - {i + 1:0>2}",
                url=f"{p.url}#{i:0>2}",
                id=f"{p.url}#{i:0>2}",
                content=f"Spezzone {i + 1} di {(len(files))}\nDa {p.mp3}",
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
    with open(
        path.join(outdir, "chiacchiericcio.xml"), "w", encoding="utf8"
    ) as text_file:
        text_file.write(feedendum.to_rss_string(feed))
    return feed


def make_site(p: Puntata, files: list[str], outdir: str) -> bool:
    if not NETLIFY_TOKEN:
        return False
    now = datetime.now()
    with open("index.html", "r", encoding="utf8") as text_file:
        template = text_file.read()

    templated = (
        template.replace("{{title}}", p.title)
        .replace("{{update}}", now.strftime("%Y-%m-%d %H:%M"))
        .replace("{{plength}}", f"n{len(files)}")
    )

    with open(path.join(outdir, "index.html"), "w", encoding="utf8") as text_file:
        text_file.write(templated)
    return upload(outdir)


def upload(outdir: str) -> bool:
    if not NETLIFY_TOKEN:
        print("NETLIFY not set")
        return False
    shutil.make_archive("file", "zip", outdir)
    client = NetlifyClient(access_token=NETLIFY_TOKEN)
    client.create_site_deploy("13bcc1e0-974e-44ae-9cbb-361a1ae3cea2", "file.zip")
    unlink("file.zip")
    return True
