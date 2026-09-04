"""Genera deck.html: toma deck.src.html e incrusta las figuras de
reports/figures/ como data URIs (los artefactos publicados no pueden
cargar imágenes externas).

Uso:  python reports/presentacion/build.py
"""
import base64
import pathlib
import re

HERE = pathlib.Path(__file__).parent
FIG = HERE.parent / "figures"   # reports/figures

src = (HERE / "deck.src.html").read_text(encoding="utf-8")


def datauri(name: str) -> str:
    data = base64.b64encode((FIG / f"{name}.png").read_bytes()).decode("ascii")
    return f"data:image/png;base64,{data}"


def repl(m: re.Match) -> str:
    name = m.group(1)
    return f'src="{datauri(name)}" data-img="{name}"'


out, n = re.subn(r'data-img="([a-z0-9_]+)"', repl, src)
(HERE / "deck.html").write_text(out, encoding="utf-8")
print(f"inyectadas {n} figuras  ->  deck.html  ({len(out) / 1024:.0f} KB)")
