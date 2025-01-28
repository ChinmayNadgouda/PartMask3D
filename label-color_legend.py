# Dictionary with labels as keys and RGB values as values
"""
'exclude2',
'hook_turn',
,
,
,
'rotate',
'foot_push',
'unplug',
'plug_in',
'pinch_pull',
   'tip_push'
"""
color_dict = {
    'exclude2': (0.0, 0.0, 0.0),
    'hook_turn': (174.0, 199.0, 232.0),
    'exclude': (188.0, 189.0, 34.0),
    'hook_pull': (152.0, 223.0, 138.0),
    'key_press': (255.0, 152.0, 150.0),
    'rotate': (214.0, 39.0, 40.0),
    'foot_push': (91.0, 135.0, 229.0),
    'unplug': (31.0, 119.0, 180.0),
    'plug_in': (229.0, 91.0, 104.0),
    'pinch_pull': (247.0, 182.0, 210.0),
    'tip_push': (91.0, 229.0, 110.0),
}
# HTML template
html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Color Display</title>
    <style>
        .color-box {{
            display: flex;
            align-items: center;
            margin: 5px 0;
        }}
        .box {{
            width: 50px;
            height: 20px;
            margin-right: 10px;
        }}
        .label {{
            font-family: Arial, sans-serif;
            font-size: 14px;
        }}
    </style>
</head>
<body>
    <h1>Color Display</h1>
    {content}
</body>
</html>
"""

# Generate color blocks with labels
color_blocks = ""
for label, rgb in color_dict.items():
    rgb_str = f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"
    color_blocks += f"""
    <div class="color-box">
        <div class="box" style="background-color: {rgb_str};"></div>
        <div class="label">{label} - {rgb_str}</div>
    </div>
    """

# Combine the template with the generated content
html_content = html_content.format(content=color_blocks)

# Save to an HTML file
with open("color_display.html", "w") as file:
    file.write(html_content)

print("HTML file has been generated: color_display.html")