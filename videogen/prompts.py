"""Ref2VA editing prompts in H3's structured Context-IR layout.

Modes (the proposal's main ablation, plus a fidelity control):
  identity     re-generate the source unchanged; measures how far Ref2VA drifts from the real clip
  object       swap the object only; the hand and arm motion must stay exactly as in the source
  object_hand  swap the object and let the grasp (finger pose / aperture) adapt to the new geometry,
               while keeping the arm trajectory, timing, pick and place locations
"""

MODES = ("identity", "object", "object_hand")

_CAMERA = (
    "<Video 1> (source video editing): fully_preserved - the exact same static side-view camera, "
    "framing, focal length, lighting, tabletop, wall and background objects, with no camera motion, "
    "no cuts and no added objects or people."
)

_PHYSICS = (
    "{Obj} is a single rigid object: it never deforms, bends, changes size or changes appearance over "
    "time. It rests stably on the table when not held, and only moves while the fingers are in contact "
    "with it. The fingers wrap around its surface without passing through it, and it does not slip, "
    "float or teleport. Its shadow on the table moves consistently with it."
)


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def build_prompt(clip: dict, mode: str, variant: dict | None = None) -> str:
    """clip: entry from configs/videogen/sources.json. variant: one of the clip's variants (unused for identity).

    The full object description is used where the object is introduced; the short `name` elsewhere.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if mode != "identity" and variant is None:
        raise ValueError(f"mode {mode!r} needs a variant")

    hand = clip.get("hand", "right")
    src_desc, src_name = clip["object"], clip.get("name", "the object")
    if mode == "identity":
        new_desc, new_name = src_desc, src_name
    else:
        new_desc, new_name = variant["description"], variant.get("name", "the new object")
    action = clip["action"].replace("the object", new_name)

    subjects = (
        f"<Subject 1> is the person's {hand} arm and hand in <Video 1>.\n"
        f"<Subject 2> is {src_desc} on the table in <Video 1>.\n"
        "<Video 1> is the source video for the editing task."
    )

    if mode == "identity":
        summary = (
            "[video editing] The target video is a faithful reproduction of <Video 1> with no changes: "
            f"<Subject 1> {action}."
        )
        retention = (
            "<Subject 1>: fully_preserved - identical arm and hand trajectory, timing, wrist orientation and "
            "finger poses in every frame.\n"
            "<Subject 2>: fully_preserved - same object, size, color and position in every frame.\n"
            f"{_CAMERA}"
        )
        grasp = f"The hand grasps {new_name} exactly as in <Video 1>."
    elif mode == "object":
        summary = (
            f"[video editing] The target video is an edited version of <Video 1> in which <Subject 2> is replaced "
            f"by {new_desc}. Only the object changes; the person's motion is identical."
        )
        retention = (
            "<Subject 1>: fully_preserved - identical arm and hand trajectory, timing, wrist orientation and "
            "finger poses in every frame, including the moments of grasp and release.\n"
            f"<Subject 2>: replaced - {new_name} takes the place of {src_name}: it starts at the same spot, is "
            "carried along exactly the same path, and is set down at the same places and times.\n"
            f"{_CAMERA}"
        )
        grasp = f"The fingers close on {new_name} at the same moments and with the same hand pose as in <Video 1>."
    else:  # object_hand
        summary = (
            f"[video editing] The target video is an edited version of <Video 1> in which <Subject 2> is replaced "
            f"by {new_desc}, and the hand's grasp naturally adapts to its size and shape."
        )
        retention = (
            "<Subject 1>: partially_preserved - the same arm, the same reach, lift, carry and place trajectory "
            "and the same timing as in <Video 1>; only the finger configuration, grip aperture and wrist "
            f"angle near the object change so that the hand forms a stable, natural grasp on {new_name}.\n"
            f"<Subject 2>: replaced - {new_name} takes the place of {src_name}: it starts at the same spot, is "
            "carried along the same path, and is set down at the same places and times.\n"
            f"{_CAMERA}"
        )
        grasp = variant.get("grasp_hint") or (
            f"The fingers open wide enough for {new_name} and close around it in a grip suited to its shape."
        )

    detail = (
        "The target video is in realistic photographic style, a continuous single shot.\n"
        "[Shot 1] A static side-view camera looks across a light wood-grain tabletop against a plain gray wall. "
        f"{_cap(new_desc)} stands on the table. <Subject 1>, the person's bare {hand} arm, enters from the right, "
        f"{action}. {grasp} {_PHYSICS.format(Obj=_cap(new_name))}"
    )

    return (
        f"subject_definitions:\n{subjects}\n\n"
        f"summary:\n{summary}\n\n"
        f"retention_analysis:\n{retention}\n\n"
        f"detailed_description:\n{detail}\n\n"
        "overall_soundscape:\nQuiet room tone with the soft sound of the object touching the table."
    )
