"""Print the pip section of a `conda env export` YAML as a requirements file, minus excluded packages.

  python scripts/daid/yml_pins.py env.yml --exclude pytorch3d kaolin > reqs.txt

Do-as-I-Do ships exact exports of its working envs (reconstruction/env/*.yml). Their conda parts use
Anaconda `defaults` builds we don't want, but the pip parts are the tested versions, so we reuse those.
Stdlib-only (no PyYAML) since it runs before the env exists.
"""

import argparse
import re
import sys


def pip_lines(path: str) -> list[str]:
    out, in_pip = [], False
    for line in open(path):
        if re.match(r"^\s*- pip:\s*$", line):
            in_pip = True
            continue
        if in_pip:
            m = re.match(r"^\s{4,}- (.+?)\s*$", line)
            if not m:
                break
            out.append(m.group(1))
    return out


def name_of(req: str) -> str:
    return re.split(r"[=<>!@ \[]", req, maxsplit=1)[0].lower().replace("_", "-")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("yml")
    ap.add_argument("--exclude", nargs="*", default=[], help="package names to drop (built separately)")
    ap.add_argument("--exclude-prefix", nargs="*", default=[], help="drop packages whose name starts with these")
    args = ap.parse_args()
    excl = {e.lower().replace("_", "-") for e in args.exclude}
    for req in pip_lines(args.yml):
        if req.startswith("--"):
            print(req)
            continue
        n = name_of(req)
        if n in excl or any(n.startswith(p) for p in args.exclude_prefix):
            print(f"# excluded: {req}", file=sys.stderr)
            continue
        print(req)


if __name__ == "__main__":
    main()
