"""``animalpose data download`` with its two network primitives stubbed out.

``urllib.request.urlopen`` (the figshare/Zenodo listings) and ``urllib.request.urlretrieve`` (the file
transfer) are replaced, so every other line runs for real: the URLs built from the ``DATASETS`` table,
the MD5 comparison that decides whether to re-download, the ``.part`` rename, the size check for the
sources with no published checksum, and the zip/tar extraction. The resulting folders are the ones
``animalpose data prepare`` reads (``<dataset>/extracted/...``, ``rat7m/mocap-<day>.mat``).
"""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
import zipfile
from pathlib import Path

import pytest

from animalpose.cli import main
from animalpose.cli.data import download as dl

EMPTY_MD5 = "d41d8cd98f00b204e9800998ecf8427e"


def md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


class Net:
    """The two calls ``download`` makes: JSON by URL, and bytes written to the destination."""

    def __init__(self, json_by_url=None, bytes_by_url=None):
        self.json_by_url, self.bytes_by_url = dict(json_by_url or {}), dict(bytes_by_url or {})
        self.apis: list[str] = []
        self.retrieved: list[tuple[str, str]] = []

    def install(self, monkeypatch) -> Net:
        monkeypatch.setattr(dl.urllib.request, "urlopen", self.urlopen)
        monkeypatch.setattr(dl.urllib.request, "urlretrieve", self.urlretrieve)
        return self

    def urlopen(self, url, timeout=60):
        assert timeout, url  # the real call always passes one
        self.apis.append(url)
        payload = io.BytesIO(json.dumps(self.json_by_url[url]).encode())

        class Response:
            def __enter__(self):
                return payload

            def __exit__(self, *exc):
                return False

        return Response()

    def urlretrieve(self, url, filename=None):
        self.retrieved.append((url, str(filename)))
        Path(filename).write_bytes(self.bytes_by_url[url])
        return str(filename), {}


def figshare_files(*files: tuple[str, bytes]) -> list[dict]:
    """The figshare API shape: name, download_url, computed_md5, size."""
    return [{"name": name, "download_url": f"https://ndownloader.figshare.com/files/{name}",
             "computed_md5": md5(data), "size": len(data)} for name, data in files]


def zenodo_files(*files: tuple[str, bytes]) -> dict:
    """The Zenodo record shape: key, links.self, "md5:<sum>", size."""
    return {"files": [{"key": name, "links": {"self": f"https://zenodo.org/records/1/files/{name}"},
                       "checksum": f"md5:{md5(data)}", "size": len(data)} for name, data in files]}


def archive_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def archive_targz(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        for name, data in files.items():
            info, data = tarfile.TarInfo(name), data
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def test_the_published_file_lists_are_read_from_the_two_apis(monkeypatch):
    files = figshare_files(("crim13.zip", b"zipzip"), ("crim13.pdf", b"pdf"))
    net = Net({"https://api.figshare.com/v2/articles/24993384/files?page_size=1000": files}).install(monkeypatch)
    kind, article, folder, pick = dl.DATASETS["crim13"]
    got = dl.figshare_files(article)
    assert kind == "figshare" and folder == "lp_crim13"
    assert net.apis == ["https://api.figshare.com/v2/articles/24993384/files?page_size=1000"]
    assert [f["name"] for f in got] == ["crim13.zip", "crim13.pdf"]
    assert got[0] == {"name": "crim13.zip", "url": "https://ndownloader.figshare.com/files/crim13.zip",
                      "md5": md5(b"zipzip"), "size": 6}
    assert pick(got[0]["name"]) and not pick(got[1]["name"])

    zen = Net({"https://zenodo.org/api/records/8250392": zenodo_files(("irodent.tar.gz", b"tt"))}).install(monkeypatch)
    kind, record, folder, pick = dl.DATASETS["irodent"]
    got = dl.zenodo_files(record)
    assert kind == "zenodo" and folder == "irodent" and pick("irodent.tar.gz") and not pick("readme.md")
    assert zen.apis == ["https://zenodo.org/api/records/8250392"]
    assert got == [{"name": "irodent.tar.gz", "url": "https://zenodo.org/records/1/files/irodent.tar.gz",
                    "md5": md5(b"tt"), "size": 2}]  # the "md5:" prefix is stripped


def test_fetch_skips_a_file_that_already_carries_the_published_md5(tmp_path, monkeypatch):
    good = tmp_path / "crim13.zip"
    good.write_bytes(b"")  # the empty file's known checksum
    assert dl.md5(good) == EMPTY_MD5
    net = Net().install(monkeypatch)
    ok = dl.fetch({"name": "crim13.zip", "url": "https://x/y.zip", "md5": EMPTY_MD5, "size": 0}, good)
    assert ok and net.retrieved == []  # nothing to redo


def test_fetch_downloads_to_a_part_file_then_renames_it(tmp_path, monkeypatch):
    net = Net(bytes_by_url={"https://x/y.zip": b"zipzip"}).install(monkeypatch)
    dest = tmp_path / "deep" / "crim13.zip"
    assert dl.fetch({"name": "y.zip", "url": "https://x/y.zip", "md5": md5(b"zipzip"), "size": 6}, dest)
    assert net.retrieved == [("https://x/y.zip", str(dest.with_suffix(".zip.part")))]
    assert dest.read_bytes() == b"zipzip" and not dest.with_suffix(".zip.part").exists()


def test_fetch_keeps_the_partial_and_says_when_the_md5_disagrees(tmp_path, monkeypatch, capsys):
    net = Net(bytes_by_url={"https://x/y.zip": b"corrupt"}).install(monkeypatch)
    dest = tmp_path / "crim13.zip"
    assert not dl.fetch({"name": "y.zip", "url": "https://x/y.zip", "md5": md5(b"zipzip"), "size": 6}, dest)
    assert not dest.exists() and dest.with_suffix(".zip.part").exists()
    assert "MD5 MISMATCH crim13.zip" in capsys.readouterr().out
    assert net.retrieved  # a real transfer was attempted


def test_unverified_sources_are_size_checked_and_their_sha_printed(tmp_path, monkeypatch, capsys):
    entry = {"name": "Snippets.zip", "url": "https://datashare/Snippets.zip", "size": 6}
    net = Net(bytes_by_url={entry["url"]: b"zipzip"}).install(monkeypatch)
    dest = tmp_path / "abode" / "Snippets.zip"
    assert dl.fetch_unverified(entry, dest)
    assert dest.read_bytes() == b"zipzip"
    printed = capsys.readouterr().out
    assert printed.startswith(f"ok {dest} (0 MB) sha256 ") and hashlib.sha256(b"zipzip").hexdigest() in printed
    assert dl.fetch_unverified(entry, dest) and len(net.retrieved) == 1  # already there, not re-fetched

    big = dict(entry, size=1_000_000)
    capsys.readouterr()
    assert not dl.fetch_unverified(big, dest)  # 6 bytes is not "within 1 % of" a megabyte
    assert "SIZE MISMATCH Snippets.zip: 6 bytes" in capsys.readouterr().out
    assert len(net.retrieved) == 1  # a size mismatch is reported, never re-downloaded


def test_archives_are_extracted_once(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    zipped, targz = raw / "a.zip", raw / "b.tar.gz"
    zipped.write_bytes(archive_zip({"crim13/val.json": b"{}", "crim13/images/0.png": b"\x89PNG"}))
    targz.write_bytes(archive_targz({"irodent/annotations.json": b"{}"}))
    dl.extract(zipped, raw / "zip_out")
    dl.extract(targz, raw / "tar_out")
    assert (raw / "zip_out/crim13/val.json").read_bytes() == b"{}"
    assert (raw / "zip_out/crim13/images/0.png").read_bytes() == b"\x89PNG"
    assert (raw / "tar_out/irodent/annotations.json").read_bytes() == b"{}"
    (raw / "zip_out" / "hand_edited").write_text("keep")
    dl.extract(zipped, raw / "zip_out")
    assert sorted(p.name for p in (raw / "zip_out").iterdir()) == ["crim13", "hand_edited"]


def test_a_figshare_archive_lands_where_prepare_reads_it(tmp_path, monkeypatch):
    archive = archive_zip({"crim13/val.json": b"[]", "crim13/images/0.png": b"x"})
    payload = figshare_files(("crim13.zip", archive), ("crim13.pdf", b"not-wanted"))
    net = Net({"https://api.figshare.com/v2/articles/24993384/files?page_size=1000": payload},
              bytes_by_url={payload[0]["download_url"]: archive}).install(monkeypatch)
    raw = tmp_path / "raw"
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "crim13", "--raw", str(raw)])
    assert stop.value.code == 0
    assert net.retrieved == [(payload[0]["download_url"], str(raw / "lp_crim13/crim13.zip.part"))]
    assert (raw / "lp_crim13/crim13.zip").exists()  # the archive stays, so a re-run verifies and skips
    # The exact folder `animalpose data prepare --crim13` opens: data/raw/lp_crim13/extracted/crim13/.
    assert (raw / "lp_crim13/extracted/crim13/val.json").read_bytes() == b"[]"
    assert not any("pdf" in url for url, _ in net.retrieved)


def test_a_second_run_verifies_the_same_file_without_downloading_again(tmp_path, monkeypatch):
    archive = archive_zip({"crim13/val.json": b"[]"})
    payload = figshare_files(("crim13.zip", archive))
    net = Net({"https://api.figshare.com/v2/articles/24993384/files?page_size=1000": payload},
              bytes_by_url={payload[0]["download_url"]: archive}).install(monkeypatch)
    raw = tmp_path / "raw"
    for _ in range(2):
        with pytest.raises(SystemExit) as stop:
            main(["data", "download", "crim13", "--raw", str(raw)])
        assert stop.value.code == 0
    assert len(net.retrieved) == 1  # the second pass only checks the MD5


def test_rat7m_takes_the_mocap_and_only_the_chunked_videos(tmp_path, monkeypatch):
    mocap = figshare_files(("mocap-s1-d1.mat", b"s1"), ("mocap-s2-d1.mat", b"s2"))
    videos = figshare_files(("mocap-videos-s2-d1-0.mp4", b"v0"), ("mocap-videos-s2-d1-122500.mp4", b"v7"),
                            ("mocap-videos-s2-d1-13000.mp4", b"not-a-chunk"), ("mocap-meta-s2-d1.json", b"{}"))
    net = Net({"https://api.figshare.com/v2/articles/13739233/files?page_size=1000": mocap,
               "https://api.figshare.com/v2/articles/13753417/files?page_size=1000": videos},
              bytes_by_url={mocap[1]["download_url"]: b"s2", videos[0]["download_url"]: b"v0",
                            videos[1]["download_url"]: b"v7"}).install(monkeypatch)
    raw = tmp_path / "raw"
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "rat7m_s2", "--raw", str(raw)])
    assert stop.value.code == 0
    assert net.apis == ["https://api.figshare.com/v2/articles/13739233/files?page_size=1000",
                        "https://api.figshare.com/v2/articles/13753417/files?page_size=1000"]
    assert sorted(Path(p).name for _, p in net.retrieved) == ["mocap-s2-d1.mat.part",
                                                              "mocap-videos-s2-d1-0.mp4.part",
                                                              "mocap-videos-s2-d1-122500.mp4.part"]
    assert {p for _, p in net.retrieved} == {str(raw / "rat7m/mocap-s2-d1.mat.part"),
                                             str(raw / "rat7m/s2-d1/mocap-videos-s2-d1-0.mp4.part"),
                                             str(raw / "rat7m/s2-d1/mocap-videos-s2-d1-122500.mp4.part")}
    # What `prepare --rat7m-s2` expects: the mocap beside the day folder full of chunked videos.
    assert (raw / "rat7m/mocap-s2-d1.mat").read_bytes() == b"s2"
    assert sorted(p.name for p in (raw / "rat7m/s2-d1").iterdir()) == ["mocap-videos-s2-d1-0.mp4",
                                                                      "mocap-videos-s2-d1-122500.mp4"]


def test_the_zenodo_tar_gz_is_picked_and_extracted(tmp_path, monkeypatch):
    record = zenodo_files(("irodent_annotations.tar.gz", archive_targz({"irodent/annotations.json": b"{}"})),
                          ("provenance.txt", b"note"))
    net = Net({"https://zenodo.org/api/records/8250392": record},
              bytes_by_url={record["files"][0]["links"]["self"]: archive_targz(
                  {"irodent/annotations.json": b"{}"})}).install(monkeypatch)
    raw = tmp_path / "raw"
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "irodent", "--raw", str(raw)])
    assert stop.value.code == 0 and len(net.retrieved) == 1
    assert net.retrieved[0][0] == record["files"][0]["links"]["self"]
    found = sorted(p.relative_to(raw).as_posix() for p in (raw / "irodent/extracted").rglob("*.json"))
    assert found == ["irodent/extracted/irodent/annotations.json"]  # prepare --irodent globs this


def test_a_failed_download_is_listed_and_exits_nonzero(tmp_path, monkeypatch, capsys):
    payload = figshare_files(("crim13.zip", b"real"))
    Net({"https://api.figshare.com/v2/articles/24993384/files?page_size=1000": payload},
        bytes_by_url={payload[0]["download_url"]: b"broken"}).install(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "crim13", "--raw", str(tmp_path / "raw")])
    assert stop.value.code == 1
    assert "FAILED: crim13.zip" in capsys.readouterr().out
    assert not (tmp_path / "raw/lp_crim13/extracted").exists()


def test_an_unknown_dataset_is_named_before_anything_is_downloaded(tmp_path, monkeypatch, capsys):
    net = Net().install(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "mirror_mouse", "not_a_dataset", "--raw", str(tmp_path / "raw")])
    assert "unknown datasets: ['not_a_dataset']" in str(stop.value)
    assert net.retrieved == [] and net.apis == []
    assert not (tmp_path / "raw").exists()


def test_direct_sources_are_size_checked_and_their_zip_extracted(tmp_path, monkeypatch, capsys):
    db = archive_zip({"full_database/mouse1.mpg": b"g"})
    Net(bytes_by_url={"https://x/full_database.zip": db, "https://x/Annotations.df": b"df",
                      "https://x/README.md": b"# doc"}).install(monkeypatch)
    monkeypatch.setitem(dl.DIRECT, "mit_homecage", [{"name": "full_database.zip", "size": len(db),
                                                     "url": "https://x/full_database.zip"}])
    monkeypatch.setitem(dl.DIRECT, "abode", [{"name": "Annotations.df", "size": 10_000,
                                              "url": "https://x/Annotations.df"},
                                             {"name": "README.md", "size": None, "url": "https://x/README.md"}])
    raw = tmp_path / "raw"
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "mit_homecage", "abode", "--raw", str(raw)])
    assert stop.value.code == 1  # ABODe's Annotations.df is 2 bytes, not 10 kB
    printed = capsys.readouterr().out
    assert "FAILED: Annotations.df" in printed
    assert (raw / "mit_homecage/extracted/full_database/mouse1.mpg").read_bytes() == b"g"
    assert (raw / "abode/README.md").read_bytes() == b"# doc"  # size None: nothing to check
    assert not (raw / "abode/extracted").exists()  # only archives get unpacked


def test_all_asks_every_api_in_the_table_once(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(dl, "DIRECT", {"mit_homecage": [], "abode": []})
    listings = {f"https://api.figshare.com/v2/articles/{a}/files?page_size=1000": []
                for a in (24993384, 24993315, dl.RAT7M_MOCAP, *dl.RAT7M_VIDEOS.values())}
    net = Net(listings | {"https://zenodo.org/api/records/8250392": {"files": []}}).install(monkeypatch)
    with pytest.raises(SystemExit) as stop:
        main(["data", "download", "--all", "--raw", str(tmp_path / "raw")])
    assert stop.value.code == 0 and net.retrieved == []
    assert "all downloads verified" in capsys.readouterr().out
    assert set(net.apis) == set(listings) | {"https://zenodo.org/api/records/8250392"}
