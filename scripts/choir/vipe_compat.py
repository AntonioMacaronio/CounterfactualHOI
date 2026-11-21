"""hf-mirror.com -> huggingface.co rewrite for the VIPE env, installed by scripts/choir/setup.sh as
site-packages/vipe_compat.py + vipe_compat.pth (loaded at interpreter startup, before torch).

CHOIR's VIPE fork downloads its torch.hub checkpoints (GroundingDINO, Depth Anything V2, Video Depth Anything,
Metric3D) from hf-mirror.com, which answers with a 308 redirect that urllib/torch.hub do not follow. The same
files live at huggingface.co under the same paths, so rewrite the host for those requests only.
"""

import urllib.request

_MIRROR, _OFFICIAL = "https://hf-mirror.com/", "https://huggingface.co/"
_urlopen = urllib.request.urlopen


def _rewrite(url):
    if isinstance(url, str) and url.startswith(_MIRROR):
        return _OFFICIAL + url[len(_MIRROR):]
    if isinstance(url, urllib.request.Request) and url.full_url.startswith(_MIRROR):
        url.full_url = _OFFICIAL + url.full_url[len(_MIRROR):]
    return url


def urlopen(url, *args, **kwargs):
    return _urlopen(_rewrite(url), *args, **kwargs)


urllib.request.urlopen = urlopen     # torch.hub does `from urllib.request import urlopen` after this runs
