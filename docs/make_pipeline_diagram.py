"""Generate docs/pipeline-{light,dark}.svg, the README pipeline diagram, with the `diagrams` library
(https://diagrams.mingrammer.com/). Needs Graphviz (`dot`) on PATH and `pip install -r docs/requirements.txt`.

    python docs/make_pipeline_diagram.py

`diagrams` writes SVG <image> elements that point at icon PNGs inside the local site-packages, which
GitHub cannot load, so every icon is inlined as a base64 data URI after rendering.
"""
import base64
import re
from pathlib import Path

from diagrams import Cluster, Diagram, Edge
from diagrams.generic.compute import Rack
from diagrams.generic.network import Switch
from diagrams.generic.storage import Storage
from diagrams.programming.flowchart import (Action, Database, Decision, LoopLimit, Preparation,
                                            PredefinedProcess)
from diagrams.programming.language import Python

HERE = Path(__file__).parent
THEMES = {
    "light": dict(bg="#fbfaf8", fg="#1b1917", sub="#736c62", edge="#9c958a",
                  cluster="#ffffff", cluster_line="#dcd7ce", accent="#2f5e96"),
    "dark": dict(bg="#0d1117", fg="#e8eef4", sub="#9aa4b0", edge="#767f8b",
                 cluster="#161b22", cluster_line="#2c333d", accent="#79aef7"),
}
FONT = "Helvetica"


def cluster_attr(c):
    return {"bgcolor": c["cluster"], "pencolor": c["cluster_line"], "fontcolor": c["sub"],
            "fontname": FONT, "fontsize": "15", "style": "rounded", "margin": "18"}


def build(theme):
    c = THEMES[theme]
    graph = {"bgcolor": c["bg"], "fontcolor": c["fg"], "fontname": FONT, "fontsize": "20",
             "pad": "0.3", "nodesep": "0.35", "ranksep": "0.5", "splines": "spline"}
    node = {"fontcolor": c["fg"], "fontname": FONT, "fontsize": "15", "height": "2.0"}
    edge = {"color": c["edge"], "fontcolor": c["sub"], "fontname": FONT, "fontsize": "15"}
    out = HERE / f"pipeline-{theme}"

    def flow(label=""):
        return Edge(color=c["edge"], label=label)

    def loop(label):
        return Edge(color=c["accent"], fontcolor=c["accent"], style="dashed", label=label,
                    constraint="false")

    with Diagram("", filename=str(out), outformat="svg", show=False, direction="LR",
                 graph_attr=graph, node_attr=node, edge_attr=edge):
        with Cluster("Mininet  ·  c2.yaml / exfil.yaml", graph_attr=cluster_attr(c)):
            attack = Rack("h2–h3 attack")
            benign = Rack("h4–h8 benign")
            switch = Switch("Open vSwitch\n10 Mbit/s")
            [attack, benign] >> flow() >> switch

        pcap = Storage("tcpdump on h1")
        features = Database("17 flow features\ndata/*.npz")
        switch >> flow() >> pcap >> flow() >> features

        with Cluster("Detectors", graph_attr=cluster_attr(c)):
            mlp = Python("MLP\nnative · PCA-8")
            vqc = Preparation("VQC\n8 qubits")
            refs = Decision("RF · XGBoost\n· rule")
        features >> flow() >> [mlp, vqc, refs]

        with Cluster("Evasion", graph_attr=cluster_attr(c)):
            sweep = LoopLimit("capture-once\nsweep")
            oracle = Decision("functional\noracle")
            score = Action("offline\nscoring")
            harden = Preparation("adversarial\ntraining")
            sweep >> flow() >> oracle >> flow() >> score >> flow() >> harden

        sweep >> loop("queries") >> switch
        [mlp, vqc, refs] >> flow() >> score
        harden >> loop("next parameter") >> sweep

    svg_path = out.with_suffix(".svg")
    svg = svg_path.read_text()

    def inline(m):
        png = Path(m.group(2))
        # the generic rack/switch/storage icons are black line art; invert them on the dark theme
        flt = ' filter="url(#invert)"' if theme == "dark" and png.parent.parent.name == "generic" else ""
        return f'{m.group(1)}"data:image/png;base64,{base64.b64encode(png.read_bytes()).decode()}"{flt}'

    svg = re.sub(r'((?:xlink:)?href=)"([^"]+\.png)"', inline, svg)
    if theme == "dark":
        defs = ('<defs><filter id="invert"><feColorMatrix type="matrix" '
                'values="-1 0 0 0 1  0 -1 0 0 1  0 0 -1 0 1  0 0 0 1 0"/></filter></defs>')
        svg = re.sub(r"(<svg\b[^>]*>)", r"\1" + defs, svg, count=1)
    svg_path.write_text(svg)
    print(f"wrote {svg_path.relative_to(HERE.parent)}")


if __name__ == "__main__":
    for t in THEMES:
        build(t)
