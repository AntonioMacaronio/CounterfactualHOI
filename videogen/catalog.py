"""Object catalog for diverse Type B swaps: concrete rigid objects with an explicit appearance.

The original ten Type B objects (B01-B10) describe geometry only, so H3 fills in neutral defaults (black box, white egg,
gray prism) and sometimes copies an object already in the scene (the "narrow-neck bottle" became the hot sauce bottle on
the counter). This catalog names everyday rigid objects that one hand can manipulate and gives each video its own
appearance: natural items get a realistic look, manufactured items a color, a material that suits them and often a
pattern or a printed label. Excluded on purpose: transparent or mirror-like objects, soft or deformable ones, and
articulated ones (the Type B prompt asks for one rigid piece, and reconstruction struggles with the first two).

  python -m videogen.catalog --base configs/videogen/real_20261119.json --per-clip 50 \
      --out configs/videogen/real_20261119_b500.json

writes a config whose clips each get `per_clip` catalog variants (mode type_b). Object types are shuffled with --seed and
dealt round-robin, so a clip never repeats a type and every type is used about equally often; each (type, appearance)
pair is unique across the whole config. Each variant carries `target` (the text for [TARGET_OBJECT_DESCRIPTION], see
group_prompts.TYPE_B_CATALOG_LINE), `cls` (short name, also the SAM3 text prompt for reconstruction), `category` and
the sampled `appearance`.
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

PALETTE = ["red", "crimson", "tomato red", "orange", "mustard yellow", "lemon yellow", "lime green", "forest green",
           "mint green", "teal", "turquoise", "sky blue", "cobalt blue", "navy blue", "lavender", "purple", "magenta",
           "hot pink", "pastel pink", "coral", "chocolate brown", "tan", "cream", "white", "black", "charcoal gray"]
PASTEL = ["pastel pink", "lavender", "mint green", "pale yellow", "baby blue", "peach", "cream", "white", "sage green"]
METAL = ["silver", "gold-toned", "copper-toned", "gunmetal gray", "rose-gold"]
WOOD = ["light oak", "dark walnut", "honey-colored pine", "reddish cherry", "pale birch", "bamboo"]
STONE = ["white marble", "black granite", "gray slate", "green jade-colored stone", "pink-veined marble", "sandstone"]

# key: (phrase with {c} for the color/tone, colors, takes a pattern)
MATERIALS = {
    "matte_plastic": ("{c} matte plastic", PALETTE, True),
    "glossy_plastic": ("{c} glossy plastic", PALETTE, True),
    "glazed_ceramic": ("{c} glazed ceramic", PALETTE, True),
    "stoneware": ("{c} matte stoneware", PALETTE, True),
    "painted_wood": ("{c} painted wooden", PALETTE, True),
    "natural_wood": ("{c} wooden", WOOD, False),
    "brushed_metal": ("{c} brushed metal", METAL, False),
    "enamel_metal": ("{c} enamel-coated metal", PALETTE, True),
    "anodized": ("{c} anodized aluminum", PALETTE, False),
    "printed_tin": ("{c} printed metal", PALETTE, True),
    "rubber": ("{c} rubber", PALETTE, True),
    "silicone": ("{c} silicone", PALETTE, True),
    "cardboard": ("{c} cardboard", PALETTE, True),
    "stone": ("{c}", STONE, False),
    "wax": ("{c} wax", PALETTE + PASTEL, True),
    "soap": ("{c}", PASTEL, True),
}

# (template with {c2}, weight)
PATTERNS = {
    "basic": [("", 5), (" with {c2} polka dots", 1), (" with thin {c2} stripes", 1), (" with a {c2} checkered pattern", 1),
              (" with a {c2} floral print", 1), (" with a {c2} zigzag pattern", 1), (" with a speckled finish", 1),
              (" with a marbled {c2} swirl", 1)],
    "label": [("", 1), (" with a printed {c2} label with bold lettering", 3), (" with {c2} stripes and a small printed logo", 1)],
    "plain": [("", 1)],
}


def N(slug, name, category, shape, size, looks):
    """Natural item: fixed list of realistic looks."""
    return dict(slug=slug, name=name, category=category, shape=shape, size=size, looks=looks)


def M(slug, name, category, shape, size, mats, pats="basic", tpl="{appearance} {name}{pattern}"):
    """Manufactured item: material from `mats` (keys of MATERIALS), color from that material, pattern set `pats`."""
    return dict(slug=slug, name=name, category=category, shape=shape, size=size, mats=mats, pats=pats, tpl=tpl)


# `shape` completes "It ..."; `size` completes "... and is about ...".
CATALOG = [
    # produce
    N("apple", "apple", "produce", "is round with a short stem", "8 cm across", ["shiny red", "bright green", "red-and-yellow streaked", "golden yellow"]),
    N("orange", "orange", "produce", "is round with a dimpled peel", "8 cm across", ["bright orange", "deep reddish-orange", "pale orange", "glossy orange"]),
    N("lemon", "lemon", "produce", "is oval with pointed ends", "7 cm long", ["bright yellow", "pale yellow", "yellow-green", "glossy lemon-yellow"]),
    N("lime", "lime", "produce", "is small and round", "5 cm across", ["bright green", "dark green", "yellow-green", "glossy green"]),
    N("peach", "peach", "produce", "is round with a soft crease down one side", "7 cm across", ["orange-and-red blushed", "golden", "rosy pink-orange", "deep red-blushed"]),
    N("pear", "pear", "produce", "is teardrop-shaped with a short stem", "10 cm tall", ["green", "yellow", "red", "golden-brown russet"]),
    N("mango", "mango", "produce", "is oval and slightly flattened", "12 cm long", ["green-and-red", "golden yellow", "orange-red blushed", "green"]),
    N("avocado", "avocado", "produce", "is pear-shaped with bumpy skin", "10 cm long", ["dark green", "nearly black", "bright green", "green-and-purple"]),
    N("potato", "potato", "produce", "is a lumpy oval", "9 cm long", ["light brown", "red-skinned", "golden", "dusty brown"]),
    N("sweet_potato", "sweet potato", "produce", "is long and tapered", "15 cm long", ["reddish-brown", "orange-brown", "purple-skinned", "pinkish"]),
    N("onion", "onion", "produce", "is round with a pointed top and papery skin", "8 cm across", ["golden-brown", "red", "white", "light brown"]),
    N("bell_pepper", "bell pepper", "produce", "is blocky with four lobes and a green stem", "9 cm tall", ["red", "yellow", "orange", "green"]),
    N("tomato", "tomato", "produce", "is round and slightly flattened with a green stem cap", "7 cm across", ["red", "orange", "yellow", "dark red"]),
    N("cucumber", "cucumber", "produce", "is long and cylindrical", "18 cm long", ["dark green", "light green", "striped green", "glossy green"]),
    N("carrot", "carrot", "produce", "is long and tapered", "17 cm long", ["bright orange", "deep orange", "purple", "yellow"]),
    N("eggplant", "eggplant", "produce", "is an elongated teardrop with a green cap", "15 cm long", ["glossy deep purple", "light purple", "white", "purple-and-white striped"]),
    N("coconut", "coconut", "produce", "is round and hairy", "11 cm across", ["brown", "dark brown", "light brown", "reddish-brown"]),
    N("pomegranate", "pomegranate", "produce", "is round with a small crown on top", "9 cm across", ["deep red", "red-and-pink", "dark crimson", "reddish-orange"]),
    N("garlic", "garlic bulb", "produce", "is round with a pointed top", "6 cm across", ["white", "purple-streaked", "off-white", "pale pink"]),
    N("zucchini", "zucchini", "produce", "is a long cylinder with a stem end", "18 cm long", ["dark green", "light green", "yellow", "speckled green"]),
    N("kiwi", "kiwi fruit", "produce", "is a small oval with fuzzy skin", "6 cm long", ["brown", "golden-brown", "dark brown", "light brown"]),
    N("grapefruit", "grapefruit", "produce", "is large and round", "11 cm across", ["yellow", "pink-blushed yellow", "pale yellow", "orange-yellow"]),
    N("corn", "corn cob", "produce", "is a cylinder covered in rows of kernels", "18 cm long", ["yellow", "white", "yellow-and-white", "multicolored"]),
    N("egg", "egg", "produce", "is oval with one narrower end", "6 cm long", ["white", "brown", "speckled cream", "pale blue"]),
    # nature
    N("pine_cone", "pine cone", "nature", "is oval with overlapping scales", "9 cm tall", ["brown", "dark brown", "light tan", "gray-brown"]),
    N("conch_shell", "conch shell", "nature", "is a spiral with a flared opening", "10 cm long", ["white-and-tan", "pink", "cream with brown speckles", "orange-banded"]),
    N("river_stone", "river stone", "nature", "is a smooth flattened oval", "8 cm long", ["gray", "black", "white", "rust-red"]),
    N("brick", "brick", "nature", "is a solid rectangular block", "20 cm long", ["red clay", "brown clay", "gray concrete", "white-painted"]),
    N("wood_offcut", "wood block", "nature", "is a rectangular block with sawn ends", "12 cm long", ["light pine", "dark walnut", "red cedar", "pale birch"]),
    # kitchen
    M("coffee_mug", "coffee mug", "kitchen", "has a cylindrical body and one side handle", "9 cm tall", ["glazed_ceramic", "stoneware", "enamel_metal", "matte_plastic"]),
    M("teacup", "teacup", "kitchen", "has a rounded flared cup and a small loop handle", "7 cm tall", ["glazed_ceramic", "stoneware"]),
    M("small_bowl", "small bowl", "kitchen", "is a shallow rounded bowl", "12 cm across", ["glazed_ceramic", "stoneware", "natural_wood", "matte_plastic"]),
    M("tumbler", "tumbler cup", "kitchen", "is a straight-sided cylindrical cup without a handle", "11 cm tall", ["brushed_metal", "enamel_metal", "matte_plastic", "stoneware"]),
    M("water_bottle", "water bottle", "kitchen", "has a cylindrical body and a screw cap", "20 cm tall", ["brushed_metal", "anodized", "matte_plastic"]),
    M("thermos", "thermos flask", "kitchen", "is a tall cylinder with a domed cup lid", "22 cm tall", ["brushed_metal", "enamel_metal"], "plain"),
    M("salt_shaker", "salt shaker", "kitchen", "is a small cylinder with a perforated dome top", "8 cm tall", ["glazed_ceramic", "brushed_metal", "natural_wood"]),
    M("pepper_grinder", "pepper grinder", "kitchen", "is a tall cylinder with a rounded knob on top", "16 cm tall", ["natural_wood", "brushed_metal", "glossy_plastic"], "plain"),
    M("spice_tin", "spice tin", "kitchen", "is a short cylindrical tin with a screw lid", "9 cm tall", ["printed_tin", "matte_plastic"], "label"),
    M("honey_pot", "honey pot", "kitchen", "has a round belly and a lid", "10 cm tall", ["glazed_ceramic", "stoneware"]),
    M("measuring_cup", "measuring cup", "kitchen", "is a small cylindrical cup with a straight handle", "7 cm tall", ["brushed_metal", "matte_plastic", "enamel_metal"], "plain"),
    M("wooden_spoon", "spoon", "kitchen", "has a long handle and an oval bowl", "25 cm long", ["natural_wood", "silicone"], "plain"),
    M("rolling_pin", "rolling pin", "kitchen", "is a cylinder with rounded handles at both ends", "25 cm long", ["natural_wood", "painted_wood", "stone"], "plain"),
    M("egg_cup", "egg cup", "kitchen", "is a small goblet shape", "6 cm tall", ["glazed_ceramic", "painted_wood", "matte_plastic"]),
    M("sugar_bowl", "sugar bowl", "kitchen", "is a round bowl with a lid and two small handles", "9 cm tall", ["glazed_ceramic", "stoneware"]),
    M("teapot", "small teapot", "kitchen", "has a round body, a spout, a handle and a lid", "12 cm tall", ["glazed_ceramic", "enamel_metal", "stoneware"]),
    M("ice_cream_scoop", "ice cream scoop", "kitchen", "has a thick handle and a round scoop", "18 cm long", ["brushed_metal", "anodized"], "plain"),
    M("citrus_juicer", "citrus juicer", "kitchen", "is a ribbed dome on a round base", "10 cm across", ["glazed_ceramic", "glossy_plastic", "natural_wood"], "plain"),
    M("kitchen_timer", "kitchen timer", "kitchen", "is egg-shaped with a twist dial", "7 cm tall", ["glossy_plastic", "brushed_metal"]),
    M("mortar", "mortar bowl", "kitchen", "is a heavy round bowl with thick walls", "10 cm across", ["stone", "stoneware"], "plain"),
    M("food_container", "food storage box", "kitchen", "is a rectangular box with a snap-on lid", "15 cm long", ["matte_plastic", "brushed_metal"], "plain"),
    M("oil_bottle", "oil bottle", "kitchen", "is a slim bottle with a narrow neck and a cork stopper", "20 cm tall", ["stoneware", "enamel_metal"]),
    # packaging
    M("soda_can", "soda can", "packaging", "is a cylinder with a pull tab on top", "12 cm tall", ["printed_tin"], "label"),
    M("soup_can", "soup can", "packaging", "is a cylinder with ridged sides", "10 cm tall", ["printed_tin"], "label"),
    M("tuna_can", "tuna can", "packaging", "is a short wide cylinder", "4 cm tall and 9 cm across", ["printed_tin"], "label"),
    M("juice_box", "juice box", "packaging", "is a small rectangular carton with a straw on the side", "10 cm tall", ["cardboard"], "label"),
    M("milk_carton", "milk carton", "packaging", "is a small carton with a gabled top", "12 cm tall", ["cardboard"], "label"),
    M("cereal_box", "cereal box", "packaging", "is a tall thin rectangular box", "18 cm tall", ["cardboard"], "label"),
    M("crisps_tube", "crisps tube", "packaging", "is a tall cylinder with a plastic lid", "20 cm tall", ["cardboard"], "label"),
    M("yogurt_cup", "yogurt cup", "packaging", "is a short tapered cup with a foil lid", "8 cm tall", ["matte_plastic"], "label"),
    M("tea_tin", "tea tin", "packaging", "is a rectangular tin with a lid", "12 cm tall", ["enamel_metal", "printed_tin"], "label"),
    M("chocolate_bar", "chocolate bar", "packaging", "is a thick flat rectangular bar in a wrapper", "15 cm long", ["cardboard"], "label"),
    M("cookie_tin", "cookie tin", "packaging", "is a round tin with a lid", "15 cm across", ["enamel_metal", "printed_tin"]),
    M("sauce_bottle", "squeeze sauce bottle", "packaging", "is a tall bottle with a flip cap", "18 cm tall", ["glossy_plastic"], "label"),
    M("nut_butter_jar", "peanut butter jar", "packaging", "is a wide cylindrical jar with a screw lid", "11 cm tall", ["matte_plastic"], "label"),
    M("candy_box", "candy box", "packaging", "is a small rectangular box", "10 cm long", ["cardboard"], "label"),
    # tools
    M("screwdriver", "screwdriver", "tools", "has a ribbed handle and a long metal shaft", "20 cm long", ["glossy_plastic", "rubber", "natural_wood"], "plain", "screwdriver with a {appearance} handle"),
    M("hammer", "hammer", "tools", "has a straight handle and a steel head", "25 cm long", ["natural_wood", "rubber", "glossy_plastic"], "plain", "hammer with a {appearance} handle"),
    M("wrench", "wrench", "tools", "is a flat bar with open jaws at one end", "18 cm long", ["brushed_metal", "enamel_metal"], "plain"),
    M("tape_measure", "tape measure", "tools", "is a rounded square case with a belt clip", "7 cm across", ["glossy_plastic", "rubber"], "plain"),
    M("duct_tape", "roll of tape", "tools", "is a thick ring", "10 cm across", ["matte_plastic"], "plain", "roll of {appearance} tape"),
    M("paintbrush", "paintbrush", "tools", "has a handle, a metal band and a flat head of bristles", "20 cm long", ["painted_wood", "natural_wood", "glossy_plastic"], "plain", "paintbrush with a {appearance} handle"),
    M("flashlight", "flashlight", "tools", "is a cylinder with a wider head", "15 cm long", ["anodized", "rubber", "glossy_plastic"], "plain"),
    M("utility_knife", "utility knife", "tools", "is a slim handle with the blade retracted", "15 cm long", ["glossy_plastic", "anodized"], "plain"),
    M("spirit_level", "spirit level", "tools", "is a long rectangular bar with a bubble vial", "23 cm long", ["anodized", "glossy_plastic"], "plain"),
    M("sanding_block", "sanding block", "tools", "is a rectangular block with a grip on top", "11 cm long", ["rubber", "matte_plastic"], "plain"),
    M("spray_paint", "spray paint can", "tools", "is a tall cylinder with a cap", "19 cm tall", ["printed_tin"], "label"),
    M("padlock", "padlock", "tools", "is a rectangular body with a curved shackle", "6 cm tall", ["brushed_metal", "enamel_metal"], "plain"),
    M("glue_bottle", "glue bottle", "tools", "is a squat bottle with a pointed nozzle cap", "12 cm tall", ["glossy_plastic"], "label"),
    M("pipe_elbow", "pipe elbow fitting", "tools", "is a short pipe bent at a right angle", "8 cm long", ["brushed_metal", "matte_plastic"], "plain"),
    # office
    M("tape_dispenser", "tape dispenser", "office", "is a weighted body holding a tape roll", "14 cm long", ["glossy_plastic", "enamel_metal"]),
    M("highlighter", "highlighter", "office", "is a flat-sided pen with a cap", "12 cm long", ["matte_plastic"], "plain"),
    M("marker", "marker pen", "office", "is a fat cylindrical pen with a cap", "14 cm long", ["matte_plastic"], "plain"),
    M("pencil_case", "hard pencil case", "office", "is a rectangular box with a lid", "20 cm long", ["matte_plastic", "enamel_metal"]),
    M("eraser", "eraser", "office", "is a small rectangular block", "6 cm long", ["rubber"]),
    M("calculator", "calculator", "office", "is a flat rectangular body with buttons", "15 cm long", ["matte_plastic"], "plain"),
    M("computer_mouse", "computer mouse", "office", "is a rounded oval body", "11 cm long", ["matte_plastic", "glossy_plastic"], "plain"),
    M("rubber_stamp", "rubber stamp", "office", "has a knob handle on a square base", "7 cm tall", ["painted_wood", "natural_wood"], "plain", "rubber stamp with a {appearance} handle"),
    M("desk_clock", "desk clock", "office", "is a round face on a small stand", "10 cm tall", ["enamel_metal", "natural_wood", "glossy_plastic"]),
    M("pencil_sharpener", "pencil sharpener", "office", "is a small rectangular block with a hole in one end", "5 cm long", ["brushed_metal", "glossy_plastic"], "plain"),
    M("board_eraser", "whiteboard eraser", "office", "is a rectangular block with a felt base", "13 cm long", ["matte_plastic", "natural_wood"], "plain"),
    M("pen_cup", "pen holder cup", "office", "is an open cylinder", "10 cm tall", ["glazed_ceramic", "brushed_metal", "natural_wood", "matte_plastic"]),
    N("paperweight", "paperweight", "office", "is a smooth dome", "8 cm across", ["polished black stone", "white marble", "green jade", "gray granite"]),
    N("puzzle_cube", "puzzle cube", "office", "is a cube covered in small square tiles", "6 cm on each side", ["classic six-color", "pastel-colored", "black-and-gold", "blue-and-white"]),
    M("glue_stick", "glue stick", "office", "is a slim cylinder with a cap", "9 cm long", ["matte_plastic"], "label"),
    # toys
    M("rubber_duck", "toy duck", "toys", "has a round body, a head and a flat beak", "9 cm tall", ["rubber", "glossy_plastic"]),
    M("toy_car", "toy car", "toys", "is a small car with four wheels", "9 cm long", ["glossy_plastic", "enamel_metal"]),
    M("toy_dinosaur", "toy dinosaur", "toys", "is a standing four-legged dinosaur figure", "12 cm long", ["matte_plastic", "rubber"]),
    M("toy_brick", "toy building brick", "toys", "is a large rectangular brick with round studs on top", "8 cm long", ["glossy_plastic"], "plain"),
    M("toy_block", "toy block", "toys", "is a cube", "6 cm on each side", ["painted_wood", "natural_wood"]),
    M("big_die", "large die", "toys", "is a cube with rounded corners and dots on each face", "6 cm on each side", ["matte_plastic", "natural_wood", "glossy_plastic"], "plain"),
    M("chess_king", "chess king", "toys", "is a tall turned piece with a cross on top", "10 cm tall", ["natural_wood", "painted_wood", "glossy_plastic", "stone"], "plain"),
    M("toy_bowling_pin", "toy bowling pin", "toys", "is a bottle shape with a narrow neck", "15 cm tall", ["glossy_plastic", "painted_wood"]),
    M("toy_robot", "toy robot", "toys", "is a boxy figure with arms and legs", "12 cm tall", ["glossy_plastic", "printed_tin"]),
    M("spinning_top", "spinning top", "toys", "is a cone with a short stem on top", "7 cm tall", ["painted_wood", "glossy_plastic", "brushed_metal"]),
    M("yo_yo", "yo-yo", "toys", "is two discs joined at the center", "6 cm across", ["glossy_plastic", "painted_wood"]),
    M("toy_train", "toy train engine", "toys", "is a small locomotive with wheels and a chimney", "12 cm long", ["painted_wood", "glossy_plastic"]),
    M("toy_boat", "toy boat", "toys", "is a small boat with a cabin", "14 cm long", ["painted_wood", "glossy_plastic"]),
    M("nesting_doll", "nesting doll", "toys", "is a rounded doll shape with a painted face", "12 cm tall", ["painted_wood"]),
    M("toy_rocket", "toy rocket", "toys", "is a cylinder with fins and a pointed nose", "15 cm long", ["glossy_plastic", "painted_wood"]),
    # sports
    N("tennis_ball", "tennis ball", "sports", "is a fuzzy sphere with a curved white seam", "7 cm across", ["optic yellow", "bright green", "pink", "orange"]),
    N("baseball", "baseball", "sports", "is a leather sphere with curved stitching", "7 cm across", ["white with red stitching", "worn off-white with red stitching", "blue-and-white", "yellow with red stitching"]),
    N("golf_ball", "golf ball", "sports", "is a small dimpled sphere", "4 cm across", ["white", "neon yellow", "orange", "pink"]),
    N("cricket_ball", "cricket ball", "sports", "is a hard leather sphere with a raised seam", "7 cm across", ["red", "white", "pink", "worn dark red"]),
    N("hockey_puck", "hockey puck", "sports", "is a flat rubber disc", "8 cm across", ["black", "orange", "blue", "red"]),
    N("ping_pong_ball", "table tennis ball", "sports", "is a small light hollow sphere", "4 cm across", ["white", "orange", "yellow", "pink"]),
    M("dumbbell", "small dumbbell", "sports", "is a short bar with hexagonal weights at both ends", "16 cm long", ["rubber", "enamel_metal", "brushed_metal"], "plain"),
    M("bocce_ball", "bocce ball", "sports", "is a heavy solid sphere", "10 cm across", ["enamel_metal", "glossy_plastic"]),
    # bathroom
    M("soap_bar", "bar of soap", "bathroom", "is a rounded rectangular bar", "9 cm long", ["soap"], "basic", "{appearance} bar of soap{pattern}"),
    M("shampoo", "shampoo bottle", "bathroom", "is an oval bottle with a flip cap", "18 cm tall", ["glossy_plastic", "matte_plastic"], "label"),
    M("deodorant", "deodorant stick", "bathroom", "is an oval cylinder with a cap", "13 cm tall", ["glossy_plastic", "matte_plastic"], "label"),
    M("lotion_pump", "lotion pump bottle", "bathroom", "is a cylinder with a pump top", "17 cm tall", ["matte_plastic", "glossy_plastic", "stoneware"], "label"),
    M("hairbrush", "hairbrush", "bathroom", "has an oval paddle head with bristles and a handle", "22 cm long", ["glossy_plastic", "natural_wood"], "plain"),
    M("razor", "razor", "bathroom", "is a slim handle with a shaving head", "14 cm long", ["glossy_plastic", "brushed_metal"], "plain"),
    M("sunscreen", "sunscreen bottle", "bathroom", "is a flat oval bottle with a cap", "15 cm tall", ["glossy_plastic"], "label"),
    M("compact", "makeup compact", "bathroom", "is a round flat case", "7 cm across", ["glossy_plastic", "brushed_metal"]),
    # household
    M("spray_bottle", "spray bottle", "household", "is a bottle with a trigger sprayer", "22 cm tall", ["glossy_plastic", "matte_plastic"], "label"),
    M("scrub_brush", "scrub brush", "household", "is a block with stiff bristles underneath", "12 cm long", ["natural_wood", "glossy_plastic"], "plain"),
    M("candle", "pillar candle", "household", "is a thick cylinder with a wick", "10 cm tall", ["wax"]),
    M("succulent_pot", "potted succulent", "household", "is a small pot holding a green succulent", "9 cm tall", ["glazed_ceramic", "stoneware"], "basic", "small {appearance} pot{pattern} holding a green succulent"),
    M("bud_vase", "small vase", "household", "has a bulbous body and a narrow neck", "15 cm tall", ["glazed_ceramic", "stoneware"]),
    M("tv_remote", "TV remote", "household", "is a long flat bar with buttons", "17 cm long", ["matte_plastic", "glossy_plastic"], "plain"),
    M("d_battery", "D battery", "household", "is a thick cylinder with a terminal bump", "6 cm long", ["printed_tin"], "label", "large D-size battery with a {appearance} casing{pattern}"),
    M("door_stop", "door stop", "household", "is a wedge", "12 cm long", ["rubber", "natural_wood"], "plain"),
    M("air_freshener", "air freshener can", "household", "is a tall slim cylinder with a spray cap", "20 cm tall", ["printed_tin"], "label"),
    M("dish_soap", "dish soap bottle", "household", "is a tall bottle with a flip-top cap", "20 cm tall", ["glossy_plastic"], "label"),
    M("piggy_bank", "piggy bank", "household", "is a pig shape with a coin slot on its back", "12 cm long", ["glazed_ceramic", "glossy_plastic", "enamel_metal"]),
    M("hand_bell", "hand bell", "household", "is a bell with a wooden handle", "12 cm tall", ["brushed_metal", "enamel_metal"], "plain"),
    M("trophy", "small trophy", "household", "is a cup on a square base", "15 cm tall", ["brushed_metal"], "plain"),
    # electronics
    M("smartphone", "smartphone", "electronics", "is a thin rectangular slab", "15 cm long", ["silicone", "glossy_plastic"], "basic", "smartphone in a {appearance} case{pattern}"),
    M("speaker", "portable speaker", "electronics", "is a squat cylinder with a fabric grille", "10 cm tall", ["rubber", "matte_plastic"], "plain"),
    M("power_bank", "power bank", "electronics", "is a flat rounded rectangle", "12 cm long", ["anodized", "matte_plastic"], "plain"),
    M("game_controller", "game controller", "electronics", "is a curved body with two grips and buttons", "15 cm wide", ["matte_plastic", "glossy_plastic"], "plain"),
    M("camera", "compact camera", "electronics", "is a small boxy body with a round lens", "11 cm wide", ["matte_plastic", "brushed_metal"], "plain"),
    M("alarm_clock", "alarm clock", "electronics", "is a round body with two bells on top", "12 cm tall", ["enamel_metal", "glossy_plastic"], "plain"),
    # decor
    M("owl_figurine", "owl figurine", "decor", "is a sitting owl shape", "10 cm tall", ["glazed_ceramic", "natural_wood", "stone"]),
    M("cat_figurine", "cat figurine", "decor", "is a sitting cat shape", "11 cm tall", ["glazed_ceramic", "stoneware", "stone"]),
    M("candle_holder", "candle holder", "decor", "is a short column with a cup on top", "9 cm tall", ["brushed_metal", "glazed_ceramic", "natural_wood"]),
    M("clay_pot", "small clay pot", "decor", "is a round pot with a rolled rim", "9 cm tall", ["stoneware", "glazed_ceramic"]),
    # shapes
    M("wood_cylinder", "cylinder block", "shapes", "is a solid cylinder", "8 cm tall", ["painted_wood", "natural_wood"]),
    M("wood_cone", "cone block", "shapes", "is a solid cone with a flat base", "9 cm tall", ["painted_wood"]),
    M("stacking_ring", "stacking ring", "shapes", "is a thick ring", "8 cm across", ["painted_wood", "glossy_plastic", "rubber"]),
    M("star_block", "star block", "shapes", "is a thick five-pointed star", "9 cm across", ["painted_wood", "glossy_plastic"]),
    M("arch_block", "arch block", "shapes", "is a rectangular block with a half-round cutout", "10 cm long", ["painted_wood"]),
    M("hex_block", "hexagonal prism block", "shapes", "is a hexagonal prism with flat ends", "8 cm long", ["painted_wood", "brushed_metal"]),
    M("solid_sphere", "solid ball", "shapes", "is a solid sphere", "7 cm across", ["painted_wood", "natural_wood", "stone"]),
    M("triangle_block", "triangular prism block", "shapes", "is a triangular prism with flat ends", "9 cm long", ["painted_wood", "glossy_plastic"]),
]


def _article(phrase: str) -> str:
    return ("An " if phrase[:1].lower() in "aeio" else "A ") + phrase


def _pick_weighted(rng: random.Random, options):
    total = sum(w for _, w in options)
    x = rng.uniform(0, total)
    for value, w in options:
        x -= w
        if x <= 0:
            return value
    return options[-1][0]


def sample_appearance(entry: dict, rng: random.Random) -> dict:
    if "looks" in entry:
        return {"look": rng.choice(entry["looks"])}
    mat = rng.choice(entry["mats"])
    phrase, colors, patterned = MATERIALS[mat]
    color = rng.choice(colors)
    pattern, c2 = "", None
    if patterned:
        pattern = _pick_weighted(rng, PATTERNS[entry["pats"]])
        if "{c2}" in pattern:
            c2 = rng.choice([c for c in PALETTE if c != color])
    return {"material": mat, "color": color, "pattern": pattern.format(c2=c2) if c2 else pattern, "accent": c2}


def describe(entry: dict, appearance: dict) -> str:
    if "look" in appearance:
        head = f"{appearance['look']} {entry['name']}"
    else:
        phrase = MATERIALS[appearance["material"]][0].format(c=appearance["color"])
        head = entry["tpl"].format(appearance=phrase, name=entry["name"], pattern=appearance["pattern"])
    return f"{_article(head)}. It {entry['shape']} and is about {entry['size']}."


def build(clip_ids: list[str], per_clip: int, seed: int) -> dict[str, list[dict]]:
    if per_clip > len(CATALOG):
        raise ValueError(f"per_clip={per_clip} exceeds the catalog size {len(CATALOG)} (a clip would repeat a type)")
    order = list(CATALOG)
    random.Random(seed).shuffle(order)
    uses = Counter()
    seen = {}  # slug -> set of descriptions already used
    out = {}
    for k, clip in enumerate(clip_ids):
        variants = []
        for j in range(per_clip):
            entry = order[(k * per_clip + j) % len(order)]
            uses[entry["slug"]] += 1
            n = uses[entry["slug"]]
            rng = random.Random(f"{seed}:{entry['slug']}:{n}")
            used = seen.setdefault(entry["slug"], set())
            for _ in range(200):  # prefer an appearance this type hasn't had yet
                app = sample_appearance(entry, rng)
                desc = describe(entry, app)
                if desc not in used:
                    break
            used.add(desc)
            variants.append({"id": f"{entry['slug']}_{n}", "mode": "type_b", "category": entry["category"],
                             "cls": entry["name"], "target": desc, "appearance": app})
        out[clip] = variants
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", type=Path, required=True, help="config whose clips to reuse (file, task, object, hand)")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--per-clip", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    base = json.loads(args.base.read_text())
    clip_ids = list(base["clips"])
    variants = build(clip_ids, args.per_clip, args.seed)
    clips = {c: {**{k: v for k, v in base["clips"][c].items() if k != "variants"}, "variants": variants[c]}
             for c in clip_ids}
    cfg = {"_comment": (f"Diverse Type B swaps from videogen/catalog.py (python -m videogen.catalog --base {args.base} "
                        f"--per-clip {args.per_clip} --seed {args.seed}): {len(CATALOG)} object types dealt round-robin "
                        f"after a seeded shuffle, {args.per_clip} distinct types per clip, one appearance per variant, "
                        "every (type, appearance) unique. 'target' fills [TARGET_OBJECT_DESCRIPTION] together with "
                        "group_prompts.TYPE_B_CATALOG_LINE; 'cls' is the short object name."),
           "clips": clips}
    args.out.write_text(json.dumps(cfg, indent=1) + "\n")
    n = sum(len(v) for v in variants.values())
    descs = {v["target"] for vs in variants.values() for v in vs}
    print(f"{n} variants over {len(clip_ids)} clips from {len(CATALOG)} types; {len(descs)} unique descriptions -> {args.out}")


if __name__ == "__main__":
    main()
