"""General-purpose media downloader (yt-dlp wrapper).

NOT THE CORPUS PATH. The LightSem corpus is AIST Dance DB / AIST++, obtained
under its research licence and already on disk (see ``lightsem.data.aist``).

This module exists so an arbitrary URL can be fetched for ad-hoc inspection —
e.g. checking a reference lighting video. Scraping a training corpus from
YouTube was considered and rejected on two independent grounds:

1. Legal — violates YouTube ToS and sits in unresolved choreography +
   footage copyright.
2. Unnecessary — AIST supplies the same content (same dancers, same music,
   multi-view, ground-truth 3D keypoints) under a research licence.

Cookie extraction from the Brave profile on this host is a CLOSED DEAD END:
``Local State`` has ``os_crypt = {"portal"}`` with no ``encrypted_key`` (Brave
uses the GNOME keyring via the XDG secrets portal), so yt-dlp reports
"Extracted 0 cookies from brave (961 could not be decrypted)". Decryption
needs an unlocked keyring in user ``kirill``'s session bus. Public videos
download fine without cookies.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


class DownloaderUnavailable(RuntimeError):
    """yt-dlp is not installed / not on PATH."""


def ytdlp_path() -> str:
    exe = shutil.which("yt-dlp")
    if exe is None:
        raise DownloaderUnavailable("yt-dlp not found on PATH")
    return exe


def download(
    url: str,
    out_dir: str | Path,
    audio_only: bool = False,
    cookie_file: str | Path | None = None,
    template: str = "%(title)s.%(ext)s",
    extra_args: list[str] | None = None,
    timeout: int = 900,
) -> subprocess.CompletedProcess:
    """Download an arbitrary URL. Returns the completed process (not parsed).

    Parameters
    ----------
    audio_only : extract audio to wav (useful for feeding the audio stage).
    cookie_file : optional Netscape-format cookie file. Browser cookie
        extraction is deliberately NOT offered — see the module docstring.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [ytdlp_path(), "-o", str(out_dir / template)]
    if audio_only:
        cmd += ["-x", "--audio-format", "wav"]
    if cookie_file is not None:
        cmd += ["--cookies", str(cookie_file)]
    if extra_args:
        cmd += list(extra_args)
    cmd.append(url)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
