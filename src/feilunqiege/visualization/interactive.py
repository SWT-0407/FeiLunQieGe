"""生成少量采样点可点击的离线 HTML 演示。

这个模块只负责把已有计算结果转换成查看器数据，不重新进行碰撞判定。
第一版故意只导出少量采样点，先验证交互方式；后续确认界面后再扩展到全部点。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def export_interactive_demo_html(
    sampled_curve: Any,
    decisions: Any,
    output_path: str | Path,
    *,
    radius: float,
    delta: float,
    units: str = "normalized",
    max_demo_points: int = 8,
) -> Path:
    """导出少量可点击采样点的离线 HTML。

    左侧是根部线的 X-Z 投影，橙色圆点是本次演示实际导出的可点击点；
    右侧根据点击结果绘制该点局部法平面内的候选中心和可行圆弧。
    所有局部坐标都由 Python 计算结果提供，网页只负责显示和交互。
    """

    if max_demo_points < 1:
        raise ValueError("max_demo_points 必须至少为 1")
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    # 将算法结果转换为只读数组，保证导出过程不改变碰撞判定数据。
    points = np.asarray(sampled_curve.points, dtype=float)
    branch_ids = np.asarray(sampled_curve.branch_ids, dtype=np.int64)
    arc_lengths = np.asarray(sampled_curve.arc_lengths, dtype=float)
    candidates = decisions.candidates
    if candidates is None:
        raise ValueError("CollisionResult 缺少 candidates，无法生成交互演示")
    feasible = np.asarray(decisions.feasible, dtype=bool)
    point_indices = np.asarray(candidates.point_indices, dtype=np.int64)
    radial_values = np.asarray(candidates.radial_values, dtype=float)
    angles = np.asarray(candidates.angles, dtype=float)
    clearance = np.asarray(decisions.clearance, dtype=float)

    # 均匀选取少量点，并优先保留一个可行率接近 50% 的解释点。
    selected_indices = _choose_demo_points(point_indices, feasible, len(points), max_demo_points)
    point_payload = []
    for point_index in selected_indices:
        mask = point_indices == point_index
        point_payload.append(
            {
                "index": int(point_index),
                "position": [float(value) for value in points[point_index]],
                "arc_length": float(arc_lengths[point_index]),
                "candidates": {
                    "rho": [float(value) for value in radial_values[mask]],
                    "theta": [float(value) for value in angles[mask]],
                    "feasible": [bool(value) for value in feasible[mask]],
                    "clearance": [float(value) for value in clearance[mask]],
                },
            }
        )

    # 仅内嵌根部线和少量候选，避免把全部三角网格及全部候选塞入浏览器。
    payload = {
        "units": units,
        "radius": float(radius),
        "delta": float(delta),
        "root_line_xz": [[float(point[0]), float(point[2])] for point in points],
        "root_line_branch_ids": [int(value) for value in branch_ids],
        "selected_points": point_payload,
    }
    html = _render_html(payload)
    output.write_text(html, encoding="utf-8")
    return output


def _choose_demo_points(
    point_indices: np.ndarray,
    feasible: np.ndarray,
    point_count: int,
    limit: int,
) -> np.ndarray:
    """均匀抽样并加入一个部分可行点，确保演示同时展示红灰两类候选。"""

    available = np.arange(point_count, dtype=np.int64)
    if point_count <= limit:
        return available

    # 先从各位置均匀选点，避免演示只集中在根部线的一小段。
    positions = np.linspace(0, point_count - 1, limit, dtype=np.int64)
    selected = set(int(value) for value in positions)

    # 如果存在部分可行点，替换距离均匀位置最近的一个点。
    ratios = []
    for point_index in available:
        mask = point_indices == point_index
        ratio = float(feasible[mask].mean()) if mask.any() else 0.0
        if 0.0 < ratio < 1.0:
            ratios.append((abs(ratio - 0.5), int(point_index)))
    if ratios:
        representative = min(ratios)[1]
        selected.add(representative)
        if len(selected) > limit:
            remove = min(selected, key=lambda value: abs(value - representative) if value != representative else -1)
            if remove == representative:
                remove = max(selected - {representative})
            selected.remove(remove)
    return np.asarray(sorted(selected), dtype=np.int64)


def _render_html(payload: dict[str, Any]) -> str:
    """把紧凑 JSON 数据嵌入一个无需服务器的 HTML 文件。"""

    data = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    # HTML 内不使用外部库；浏览器双击文件即可打开，适合初次验收。
    return f'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FeiLunQieGe interactive demo</title>
  <style>
    :root {{ color-scheme: light; font-family: system-ui, "Microsoft YaHei", sans-serif; }}
    body {{ margin: 0; padding: 18px; color: #1f2937; background: #f8fafc; }}
    h1 {{ margin: 0 0 6px; font-size: 20px; }}
    .subtitle {{ margin: 0 0 14px; color: #475569; font-size: 13px; }}
    .layout {{ display: grid; grid-template-columns: minmax(360px, 1fr) minmax(360px, 1fr); gap: 14px; }}
    .panel {{ background: #ffffff; border: 1px solid #cbd5e1; padding: 10px; }}
    .panel h2 {{ margin: 0 0 6px; font-size: 15px; }}
    svg {{ width: 100%; height: auto; display: block; background: #ffffff; }}
    .axis {{ stroke: #94a3b8; stroke-width: 1; }}
    .root-line {{ fill: none; stroke: #2563eb; stroke-width: 2.2; }}
    .demo-point {{ fill: #f97316; stroke: #7c2d12; stroke-width: 1.5; cursor: pointer; }}
    .demo-point.selected {{ fill: #dc2626; stroke: #111827; stroke-width: 2.5; }}
    .candidate-feasible {{ fill: #dc2626; }}
    .candidate-rejected {{ stroke: #6b7280; stroke-width: 1.1; }}
    .band-outer {{ fill: none; stroke: #111827; stroke-width: 1.5; }}
    .band-inner {{ fill: none; stroke: #2563eb; stroke-width: 1.3; stroke-dasharray: 5 4; }}
    .feasible-arc {{ fill: none; stroke: #ea580c; stroke-width: 5; stroke-linecap: round; }}
    .detail {{ margin-top: 8px; min-height: 42px; font-size: 13px; line-height: 1.5; }}
    .point-list {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0 0; }}
    button {{ border: 1px solid #94a3b8; background: #f8fafc; color: #1f2937; padding: 5px 8px; cursor: pointer; }}
    button[aria-pressed="true"] {{ background: #fee2e2; border-color: #dc2626; }}
    .legend {{ font-size: 12px; color: #475569; margin-top: 6px; }}
    @media (max-width: 800px) {{ .layout {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <h1>砂轮可行中心交互演示</h1>
  <p class="subtitle">本页只导出少量代表性采样点。单位：<span id="units"></span>。红色候选为可行中心，灰色候选为碰撞筛除中心。</p>
  <div class="layout">
    <section class="panel">
      <h2>根部线投影：点击橙色采样点</h2>
      <svg id="rootSvg" viewBox="0 0 640 420" role="img" aria-label="根部线 X-Z 投影和可点击采样点"></svg>
      <div id="pointButtons" class="point-list" aria-label="采样点选择"></div>
      <div class="legend">蓝线：root_line；橙色点：本次演示可点击的采样点；红色点：当前选中点。</div>
    </section>
    <section class="panel">
      <h2>当前点的局部法平面放大图</h2>
      <svg id="localSvg" viewBox="0 0 640 420" role="img" aria-label="当前采样点的可行半径区域"></svg>
      <div id="details" class="detail" aria-live="polite"></div>
      <div class="legend">黑线：rho=r；蓝色虚线：rho=r-delta；红点：可行中心；灰色叉号：不可行中心；橙色粗线：外半径可行圆弧。</div>
    </section>
  </div>
  <script>
    const DATA = {data};
    const NS = "http://www.w3.org/2000/svg";
    let selected = 0;

    function svg(tag, attrs = {{}}) {{
      const node = document.createElementNS(NS, tag);
      for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
      return node;
    }}

    function extent(values) {{
      return [Math.min(...values), Math.max(...values)];
    }}

    function scale(value, domain, start, end) {{
      const span = domain[1] - domain[0] || 1;
      return start + (value - domain[0]) / span * (end - start);
    }}

    function drawRoot() {{
      const root = document.getElementById("rootSvg");
      root.replaceChildren();
      const xValues = DATA.root_line_xz.map(p => p[0]);
      const zValues = DATA.root_line_xz.map(p => p[1]);
      const xDomain = extent(xValues), zDomain = extent(zValues);
      const left = 54, right = 620, top = 20, bottom = 370;
      root.appendChild(svg("line", {{x1:left, y1:bottom, x2:right, y2:bottom, class:"axis"}}));
      root.appendChild(svg("line", {{x1:left, y1:top, x2:left, y2:bottom, class:"axis"}}));
      let run = [];
      function flush() {{
        if (run.length > 1) {{
          const path = run.map((p, i) => (i ? "L" : "M") + scale(p[0], xDomain, left, right) + " " + scale(p[1], zDomain, bottom, top)).join(" ");
          root.appendChild(svg("path", {{d:path, class:"root-line"}}));
        }}
        run = [];
      }}
      let lastBranch = DATA.root_line_branch_ids[0];
      DATA.root_line_xz.forEach((point, i) => {{
        if (DATA.root_line_branch_ids[i] !== lastBranch) {{ flush(); lastBranch = DATA.root_line_branch_ids[i]; }}
        run.push(point);
      }});
      flush();
      DATA.selected_points.forEach((point, i) => {{
        const [x, z] = point.position;
        const circle = svg("circle", {{cx:scale(x, xDomain, left, right), cy:scale(z, zDomain, bottom, top), r: i === selected ? 8 : 6, class:"demo-point" + (i === selected ? " selected" : "")}});
        circle.setAttribute("tabindex", "0");
        circle.setAttribute("aria-label", "采样点 " + point.index);
        circle.addEventListener("click", () => selectPoint(i));
        circle.addEventListener("keydown", event => {{ if (event.key === "Enter" || event.key === " ") selectPoint(i); }});
        root.appendChild(circle);
        const label = svg("text", {{x:scale(x, xDomain, left, right) + 8, y:scale(z, zDomain, bottom, top) - 8, "font-size":"12", fill:"#7c2d12"}});
        label.textContent = String(point.index);
        root.appendChild(label);
      }});
      const xlabel = svg("text", {{x:300, y:405, "font-size":"12", fill:"#334155"}}); xlabel.textContent = "X (normalized units)"; root.appendChild(xlabel);
      const zlabel = svg("text", {{x:12, y:205, "font-size":"12", fill:"#334155", transform:"rotate(-90 12 205)"}}); zlabel.textContent = "Z (normalized units)"; root.appendChild(zlabel);
    }}

    function drawLocal() {{
      const point = DATA.selected_points[selected];
      const local = point.candidates;
      const root = document.getElementById("localSvg");
      root.replaceChildren();
      const left = 54, right = 600, top = 20, bottom = 370;
      const limit = DATA.radius * 1.35;
      // 局部法平面必须使用相同的 x/y 像素比例，否则数学圆会被显示成椭圆。
      const centerX = (left + right) / 2;
      const centerY = (top + bottom) / 2;
      const unitScale = Math.min(right - left, bottom - top) / (2 * limit);
      const sx = value => centerX + value * unitScale;
      const sy = value => centerY - value * unitScale;
      root.appendChild(svg("line", {{x1:left, y1:sy(0), x2:right, y2:sy(0), class:"axis"}}));
      root.appendChild(svg("line", {{x1:sx(0), y1:top, x2:sx(0), y2:bottom, class:"axis"}}));
      function circle(radius, className) {{ root.appendChild(svg("circle", {{cx:sx(0), cy:sy(0), r:radius * unitScale, class:className}})); }}
      circle(DATA.radius, "band-outer");
      circle(DATA.radius - DATA.delta, "band-inner");
      const outer = [];
      for (let i = 0; i < local.rho.length; i++) {{
        const x = local.rho[i] * Math.cos(local.theta[i]);
        const y = local.rho[i] * Math.sin(local.theta[i]);
        if (Math.abs(local.rho[i] - DATA.radius) < 1e-10 && local.feasible[i]) outer.push([x, y, local.theta[i]]);
        if (local.feasible[i]) root.appendChild(svg("circle", {{cx:sx(x), cy:sy(y), r:3.2, class:"candidate-feasible"}}));
        else {{
          root.appendChild(svg("line", {{x1:sx(x)-3, y1:sy(y)-3, x2:sx(x)+3, y2:sy(y)+3, class:"candidate-rejected"}}));
          root.appendChild(svg("line", {{x1:sx(x)-3, y1:sy(y)+3, x2:sx(x)+3, y2:sy(y)-3, class:"candidate-rejected"}}));
        }}
      }}
      outer.sort((a,b) => a[2] - b[2]);
      const angleStep = 2 * Math.PI / Math.max(outer.length, 1);
      outer.forEach((current, i) => {{
        const next = outer[(i + 1) % outer.length];
        const gap = (next[2] - current[2] + 2*Math.PI) % (2*Math.PI);
        if (gap <= angleStep * 1.1) root.appendChild(svg("line", {{x1:sx(current[0]), y1:sy(current[1]), x2:sx(next[0]), y2:sy(next[1]), class:"feasible-arc"}}));
      }});
      root.appendChild(svg("circle", {{cx:sx(0), cy:sy(0), r:6, fill:"#111827"}}));
      const details = document.getElementById("details");
      const feasibleCount = local.feasible.filter(Boolean).length;
      const minimum = Math.min(...local.clearance);
      details.textContent = `采样点 ${{point.index}}；弧长位置 ${{point.arc_length.toFixed(5)}}；可行候选 ${{feasibleCount}}/${{local.feasible.length}}；最小候选余量 ${{minimum.toFixed(6)}} ${{DATA.units}}`;
    }}

    function selectPoint(index) {{
      selected = index;
      document.querySelectorAll("#pointButtons button").forEach((button, i) => button.setAttribute("aria-pressed", String(i === selected)));
      drawRoot();
      drawLocal();
    }}

    function buildPointButtons() {{
      const container = document.getElementById("pointButtons");
      DATA.selected_points.forEach((point, i) => {{
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = "点 " + point.index;
        button.setAttribute("aria-pressed", String(i === selected));
        button.addEventListener("click", () => selectPoint(i));
        container.appendChild(button);
      }});
    }}

    document.getElementById("units").textContent = DATA.units;
    buildPointButtons();
    drawRoot();
    drawLocal();
  </script>
</body>
</html>
'''
