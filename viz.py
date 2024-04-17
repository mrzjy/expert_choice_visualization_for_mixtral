from collections import Counter


html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Colorful Words</title>
    <style>
        body {{
            width: 400px;
            word-wrap: break-word;
        }}
        .expert {{
            display: inline;
            padding: 5px;
            font-size: 10px;
            color: white;
        }}
{rendered_expert_style}
        .legend {{
            margin-top: 10px;
        }}
        .legend div {{
            padding: 2px;
        }}
    </style>
</head>
<body>
    <div>
        {rendered_tokens}
    </div>
    <div class="legend">
{rendered_legend}
    </div>
</body>
</html>"""


experts = {
    "1": {"color": "red"},
    "2": {"color": "green"},
    "3": {"color": "blue"},
    "4": {"color": "black"},
    "5": {"color": "purple"},
    "6": {"color": "orange"},
    "7": {"color": "pink"},
    "8": {"color": "cyan"},
}

expert_style_template = """        .expert{i} {{
            background-color: {color};
        }}"""

word_template = """<span class="expert expert{i}">{token}</span>"""

legend_template = """        <div style="color: {color};">expert{i} ({ratio:.2f}%): {color}</div>"""


def format_html(tokens, output_file="tmp.html"):
    rendered_tokens = "".join([word_template.format(token=token, i=i) for i, token in tokens])
    rendered_expert_style = "\n".join([
        expert_style_template.format(i=i, color=info["color"]) for i, info in experts.items()
    ])

    counter = Counter([str(i) for i, t in tokens])

    rendered_legend = "\n".join([
        legend_template.format(i=i, color=info["color"], ratio=counter[i]/len(tokens) * 100) for i, info in experts.items()
    ])
    content = html.format(
        rendered_expert_style=rendered_expert_style,
        rendered_tokens=rendered_tokens,
        rendered_legend=rendered_legend,
    )
    with open(output_file, "w", encoding="utf-8") as f:
        print(content, file=f)
