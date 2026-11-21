# H3 prompt: pick → in-hand rotation → place, Type A (geometry edit)

Copied from Alvin's group doc ("381V Visual Recognition Group Project" → Prompt). Used for all three geometry edits on every prp real video: 3 clips per real video, 60 total.

Fill `[GEOMETRIC_TRANSFORMATION]` with the text for one edit ID:

| ID | Category | Text to insert |
|---|---|---|
| A01 | Shorter | Reduce the object's height to 80% of its original height. Keep its width and depth unchanged. |
| A02 | Narrower | Reduce the object's width and depth to 80% of their original values. Keep its height unchanged. |
| A03 | Larger overall | Uniformly increase all object dimensions to 120% of their original values. |

## Prompt

```
Create a physically plausible counterfactual edit of the provided manipulation
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
```
