# coding: utf-8
# @Author: Wang Qingkang


import webbrowser
from pathlib import Path
from tempfile import gettempdir

from agent_layer.graph import build_graph


GRAPH_IMAGE_PATH = Path(gettempdir()) / "tender_qa_agent_graph.png"


def display_graph() -> None:
    graph = build_graph()
    graph.get_graph().draw_mermaid_png(
        output_file_path=GRAPH_IMAGE_PATH.as_posix(),
    )
    webbrowser.open(GRAPH_IMAGE_PATH.as_uri())


if __name__ == "__main__":
    display_graph()