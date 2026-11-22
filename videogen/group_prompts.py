"""MiniMax H3 prompts shared by the 381V group (Alvin's "381V Visual Recognition Group Project" doc).

Type A (geometry): same object, one geometric change. Type B (object swap): one rigid object from a fixed list.
The pick -> in-hand rotation -> place (prp) templates are copied verbatim from
antzhang-notes/prompt_prp_type_a.md and prompt_prp_type_b.md. A04/A05 extend Type A to five edits per clip
(added 2026-11-21: taller, smaller overall), mirroring A01 and A03.

Pick-and-place (pp) clips have no in-hand reorientation, so they use the same text with the rotation-specific
steps and sentences replaced by a carry (`_PP_EDITS_A`, `_PP_EDITS_B`). Each replacement must match exactly once,
so an edit to the verbatim templates fails loudly instead of silently producing a half-converted prompt.
"""

TYPE_A_EDITS = {
    "A01": ("Shorter", "Reduce the object's height to 80% of its original height. Keep its width and depth unchanged."),
    "A02": ("Narrower", "Reduce the object's width and depth to 80% of their original values. Keep its height unchanged."),
    "A03": ("Larger overall", "Uniformly increase all object dimensions to 120% of their original values."),
    "A04": ("Taller", "Increase the object's height to 120% of its original height. Keep its width and depth unchanged."),
    "A05": ("Smaller overall", "Uniformly reduce all object dimensions to 80% of their original values."),
}

# Type B's [TARGET_OBJECT_DESCRIPTION] is this line followed by the object's text.
TYPE_B_SIZE_LINE = (
    "Size the target object for manipulation by the same hand, at approximately the source object's overall "
    "scale. Keep it rigid throughout the action. Adapt the grasp and local motion to its geometry."
)

# Catalog swaps (videogen/catalog.py) fill [TARGET_OBJECT_DESCRIPTION] with a concrete object and appearance followed
# by this line instead of TYPE_B_SIZE_LINE: the object keeps its natural size rather than the source's scale, and the
# last sentence stops H3 from reusing an object already in the scene (it turned "a narrow-neck bottle" into the
# hot sauce bottle on the counter).
TYPE_B_CATALOG_LINE = (
    "Show it at its natural real-world size, rigid and in one piece, so the same hand can manipulate it. It must "
    "look different from every other object in the scene. Adapt the grasp and local motion to its geometry."
)

TYPE_B_OBJECTS = {
    "B01": ("Cylindrical can", "A closed cylindrical beverage can with straight sides, a flat circular base, and a metallic surface."),
    "B02": ("Narrow-neck bottle", "A capped rigid bottle with a rounded body, narrow neck, and flat base."),
    "B03": ("Handled mug", "A rigid mug with a cylindrical body, one side handle, and a flat base."),
    "B04": ("Rectangular box", "A closed rectangular box with unequal side lengths, flat faces, and slightly rounded edges."),
    "B05": ("Wide jar", "A closed, short, wide cylindrical jar with a screw-top lid and a flat base."),
    "B06": ("Spherical object", "A rigid, nearly spherical object with a small flattened base for stable placement."),
    "B07": ("Egg-shaped object", "A rigid egg-shaped object with one narrower end and a small flattened base for stable placement."),
    "B08": ("Pear-shaped object", "A rigid pear-shaped object with a broad lower body, narrow upper body, and flat base."),
    "B09": ("Triangular prism", "A solid triangular prism with slightly rounded edges and a flat face suitable for stable placement."),
    "B10": ("Hexagonal prism", "A solid hexagonal prism with six flat side faces, slightly rounded edges, and flat ends."),
}

TYPE_A_PRP = """Create a physically plausible counterfactual edit of the provided manipulation
video. Modify the source object's geometry according to the intervention
below, and adapt the interacting hand and motion so that the complete
manipulation remains physically feasible.

OBJECT INTERVENTION
[GEOMETRIC_TRANSFORMATION]

Apply the specified transformation in the object's local coordinate system,
not in image coordinates. The transformed object must have one consistent
three-dimensional shape and fixed dimensions throughout the video, including
while it rotates or becomes partially occluded.

Preserve the source object's identity, material, color, and recognizable
surface features as closely as the geometric transformation allows. Change
only the specified geometric factors. Do not introduce an additional object,
replace the object with a different category, or change its geometry during
the action.

TASK SEQUENCE
Preserve the following manipulation sequence:
1. The manipulating hand approaches the object at its source starting location.
2. The fingers establish a grasp appropriate for the modified geometry.
3. The hand lifts the object fully clear of the support surface.
4. The hand reorients the object through coordinated finger motion while
   maintaining support.
5. The hand lowers the object to the source placement location.
6. The object establishes stable contact with the support surface.
7. The fingers release the object and the hand withdraws.

Preserve the source task's broad motion pattern, approximate rotation
direction, and approximate extent of reorientation. During the in-hand
rotation phase, the object must change orientation relative to the palm.
Do not replace this phase with a wrist-only rotation of an otherwise
unchanged grasp.

GEOMETRY-DEPENDENT HAND ADAPTATION
Adapt finger spread, finger flexion, thumb opposition, grasp depth, and
contact locations to the modified object. Allow coordinated finger
repositioning during in-hand rotation.

Preserve the same hand identity, handedness, anatomical structure, finger
count, and approximate anatomical dimensions. Adapt the hand configuration
rather than stretching the fingers, changing finger lengths, or resizing
the hand.

Allow changes to wrist pose, approach direction, lift height, and local arm
motion when needed for reachability, object clearance, and secure handling.
Keep these changes consistent with the source task. Do not force the
original finger poses or exact wrist trajectory when they are incompatible
with the modified geometry.

PHYSICAL INTERACTION
The object behaves as one rigid body. Its motion must be consistent with
visible hand support, contact, gravity, and interaction with the support
surface.

Establish supporting contacts before lifting. During airborne manipulation,
maintain a plausible supporting grasp using the fingers and/or palm.
Individual fingers may release and reposition, but the remaining contacts
must continue to support the object.

Show coordinated changes in finger contact and object motion. Avoid
unexplained sliding, unsupported floating, sudden orientation changes,
interpenetration, or fingers passing through the object.

Lower the object until it contacts the support surface before releasing
it. Leave it in a stable resting pose without unexplained bouncing,
drifting, or tipping.

SCENE AND TEMPORAL CONSISTENCY
Preserve the source scene, unrelated objects, support surface, lighting,
camera viewpoint, and framing. Change only the manipulated object, the
interacting hand and nearby arm motion as needed, and directly affected
occlusions, shadows, and reflections.

Keep the initial support location and final placement location consistent
with the source. Allow small changes in object center height and hand
position required by the new dimensions.

Preserve the source phase order and approximate duration. Allow modest
timing adjustments needed for a feasible grasp, rotation, and release.
Produce one continuous shot with smooth motion and no cuts.
"""

TYPE_B_PRP = """Create a physically plausible counterfactual version of the provided
manipulation video by replacing the manipulated object with the target
object below. Adapt the grasp and manipulation to the replacement object
so that the complete action remains physically feasible.

TARGET OBJECT
[TARGET_OBJECT_DESCRIPTION]

Show exactly one manipulated instance of this target object. Maintain its
identity, three-dimensional geometry, dimensions, material, color, and
surface features throughout the video, including during rotation and
partial occlusion.

The target is a rigid, single-piece object suitable for one-handed
manipulation. Do not morph it between shapes or change its dimensions
during the action.

TASK SEQUENCE
Preserve the following task:
1. Approach the target object at the source starting location.
2. Establish a grasp suited to its geometry.
3. Lift it fully clear of the support surface.
4. Reorient it within the manipulating hand.
5. Lower it to the source placement location.
6. Establish stable contact with the support surface.
7. Release the object and withdraw the hand.

Preserve the source task's phase order, broad movement pattern, and
approximate rotation direction and extent. Adapt the detailed manipulation
to the target object.

During in-hand reorientation, the target object must visibly change
orientation relative to the palm through coordinated finger motion.
Do not substitute a wrist-only rotation, a toss, a second-hand assist,
or a temporary placement on the surface.

OBJECT-SPECIFIC GRASP AND MOTION
Choose finger spread, finger flexion, thumb opposition, grasp depth, and
contact locations appropriate for the target object's dimensions and
surface geometry.

Allow the fingers to reposition during rotation. Individual contacts may
be released and re-established while other fingers and/or the palm continue
to support the object.

Adapt wrist pose, approach direction, lift height, and local arm motion
when needed for reachability, secure handling, and collision avoidance.
Preserve the source manipulation workspace and overall task intent.

Preserve the manipulating hand's identity, handedness, anatomy, finger
count, and approximate anatomical dimensions. Do not resize or deform the
hand to fit the target object.

PHYSICAL INTERACTION
Maintain plausible support throughout airborne manipulation. Couple object
translation and rotation to visible contact changes. Show contact on the
actual target surfaces rather than at the source object's former surfaces.

Avoid interpenetration, unsupported floating, unexplained sliding,
uncontrolled rolling, abrupt motion, and object motion without a plausible
supporting interaction.

Place the target in a stable resting pose at the source placement location.
Establish supporting surface contact before release. The hand must disengage
without passing through or disturbing the placed object.

SCENE AND TEMPORAL CONSISTENCY
Preserve the source scene, unrelated objects, support surface, lighting,
camera viewpoint, and framing. Change the manipulated object, the
interacting hand and nearby arm motion as needed, and directly affected
occlusions, shadows, and reflections.

Preserve the source phase order and approximate duration while allowing
modest timing changes required by the target geometry. Produce one
continuous shot with smooth motion and no cuts.
"""

# (old, new) replacements turning the prp templates into pick-and-place ones.
_PP_EDITS_A = [
    ("including\nwhile it rotates or becomes partially occluded.",
     "including\nwhile it moves or becomes partially occluded."),
    ("4. The hand reorients the object through coordinated finger motion while\n   maintaining support.\n"
     "5. The hand lowers the object to the source placement location.\n"
     "6. The object establishes stable contact with the support surface.\n"
     "7. The fingers release the object and the hand withdraws.",
     "4. The hand carries the object and lowers it to the source placement location.\n"
     "5. The object establishes stable contact with the support surface.\n"
     "6. The fingers release the object and the hand withdraws."),
    ("Preserve the source task's broad motion pattern, approximate rotation\n"
     "direction, and approximate extent of reorientation. During the in-hand\n"
     "rotation phase, the object must change orientation relative to the palm.\n"
     "Do not replace this phase with a wrist-only rotation of an otherwise\nunchanged grasp.",
     "Preserve the source task's broad motion pattern and carrying path.\n"
     "While the object is carried, keep a stable grasp: the object does not\n"
     "slip or turn within the hand."),
    ("contact locations to the modified object. Allow coordinated finger\n"
     "repositioning during in-hand rotation.",
     "contact locations to the modified object."),
    ("maintain a plausible supporting grasp using the fingers and/or palm.\n"
     "Individual fingers may release and reposition, but the remaining contacts\n"
     "must continue to support the object.",
     "maintain a plausible supporting grasp using the fingers and/or palm."),
    ("Show coordinated changes in finger contact and object motion. Avoid",
     "Couple the object's motion to the hand's motion. Avoid"),
    ("grasp, rotation, and release.", "grasp, carry, and release."),
]

_PP_EDITS_B = [
    ("including during rotation and\npartial occlusion.",
     "including while it moves and\nduring partial occlusion."),
    ("4. Reorient it within the manipulating hand.\n"
     "5. Lower it to the source placement location.\n"
     "6. Establish stable contact with the support surface.\n"
     "7. Release the object and withdraw the hand.",
     "4. Carry it and lower it to the source placement location.\n"
     "5. Establish stable contact with the support surface.\n"
     "6. Release the object and withdraw the hand."),
    ("broad movement pattern, and\napproximate rotation direction and extent.",
     "broad movement pattern, and\ncarrying path."),
    ("During in-hand reorientation, the target object must visibly change\n"
     "orientation relative to the palm through coordinated finger motion.\n"
     "Do not substitute a wrist-only rotation, a toss, a second-hand assist,\n"
     "or a temporary placement on the surface.",
     "While it is carried, the target object stays securely held and does not\n"
     "slip or turn within the hand. Do not substitute a toss, a second-hand\n"
     "assist, or a temporary placement on the surface."),
    ("Allow the fingers to reposition during rotation. Individual contacts may\n"
     "be released and re-established while other fingers and/or the palm continue\n"
     "to support the object.\n\n",
     ""),
    ("Couple object\ntranslation and rotation to visible contact changes.",
     "Couple the object's\nmotion to the hand's motion."),
]


def _apply(text: str, edits: list[tuple[str, str]]) -> str:
    for old, new in edits:
        n = text.count(old)
        if n != 1:
            raise AssertionError(f"pp edit must match the template exactly once, matched {n}x: {old[:60]!r}")
        text = text.replace(old, new)
    return text


TYPE_A_PP = _apply(TYPE_A_PRP, _PP_EDITS_A)
TYPE_B_PP = _apply(TYPE_B_PRP, _PP_EDITS_B)

TEMPLATES = {
    ("type_a", "pick_rotate_place"): TYPE_A_PRP,
    ("type_b", "pick_rotate_place"): TYPE_B_PRP,
    ("type_a", "pick_place"): TYPE_A_PP,
    ("type_b", "pick_place"): TYPE_B_PP,
}


def edit_info(mode: str, edit_id: str) -> tuple[str, str]:
    """(category, inserted text) for a Type A edit or Type B object id."""
    table = TYPE_A_EDITS if mode == "type_a" else TYPE_B_OBJECTS
    if edit_id not in table:
        raise ValueError(f"unknown {mode} id {edit_id!r}; expected one of {sorted(table)}")
    return table[edit_id]


def build_group_prompt(task: str, mode: str, edit_id: str, target: str | None = None) -> str:
    """task: 'pick_rotate_place' or 'pick_place'. mode: 'type_a' or 'type_b'. edit_id: e.g. 'A03' or 'B07'.

    target: a catalog object description (videogen/catalog.py) to use instead of looking edit_id up in the tables.
    """
    if (mode, task) not in TEMPLATES:
        raise ValueError(f"no group prompt for mode={mode!r}, task={task!r}")
    if target is not None:
        if mode != "type_b":
            raise ValueError("catalog targets are Type B object swaps")
        return TEMPLATES[mode, task].replace("[TARGET_OBJECT_DESCRIPTION]", f"{target}\n{TYPE_B_CATALOG_LINE}").strip()
    _, text = edit_info(mode, edit_id)
    if mode == "type_a":
        return TEMPLATES[mode, task].replace("[GEOMETRIC_TRANSFORMATION]", text).strip()
    return TEMPLATES[mode, task].replace("[TARGET_OBJECT_DESCRIPTION]", f"{TYPE_B_SIZE_LINE}\n{text}").strip()
