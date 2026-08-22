import re
import webcolors

# Define a smaller palette
basic_colors = {
    # Neutrals
    "black": "#000000",
    "dark_gray": "#555555",
    "gray": "#808080",
    "light_gray": "#d3d3d3",
    "white": "#ffffff",

    # Reds & Oranges
    "red": "#e53935",
    "crimson": "#b71c1c",
    "orange": "#fb8c00",
    "amber": "#ffb300",

    # Yellows
    "yellow": "#fdd835",
    "gold": "#ffd700",

    # Greens
    "lime": "#c0ca33",
    "green": "#43a047",
    "emerald": "#00a86b",

    # Cyans & Blues
    "cyan": "#00acc1",
    "teal": "#00897b",
    "sky_blue": "#42a5f5",
    "blue": "#1e88e5",
    "navy": "#283593",

    # Purples & Pinks
    "indigo": "#5e35b1",
    "purple": "#8e24aa",
    "violet": "#7c4dff",
    "magenta": "#d81b60",
    "pink": "#ec407a",

    # Browns
    "brown": "#795548",
    "tan": "#d2b48c",

    # Extras
    "olive": "#808000",
    "turquoise": "#40e0d0"
}


def color_readable(rgb_string):
    """Converts a numerical color value to a distinct name say brown
    Args:
        rgb_string (string): its of the form "rgb(40,40,40)"

    Returns:
        closest name of the color"""
    nums = list(map(int, re.findall(r"\d+", rgb_string)))
    r, g, b = nums[:3]
    requested = (r, g, b)
    min_dist = float("inf")
    closest = None
    for name, hexval in basic_colors.items():
        rc, gc, bc = webcolors.hex_to_rgb(hexval)
        dist = (rc-r)**2 + (gc-g)**2 + (bc-b)**2
        if dist < min_dist:
            min_dist = dist
            closest = name
    return closest


# print(closest_basic_color("rgb(144, 3, 252)"))  # → magenta
