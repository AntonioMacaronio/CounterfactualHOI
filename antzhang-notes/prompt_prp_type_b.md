# H3 prompt: pick → in-hand rotation → place, Type B (object swap)

Copied from Alvin's group doc ("381V Visual Recognition Group Project" → Prompt). Used for 3 of the 10 target objects per prp real video: 3 clips per real video, 60 total.

Fill `[TARGET_OBJECT_DESCRIPTION]` with this line followed by the text for one object ID:

> Size the target object for manipulation by the same hand, at approximately the source object's overall scale. Keep it rigid throughout the action. Adapt the grasp and local motion to its geometry.

| ID | Category | Text to insert |
|---|---|---|
| B01 | Cylindrical can | A closed cylindrical beverage can with straight sides, a flat circular base, and a metallic surface. |
| B02 | Narrow-neck bottle | A capped rigid bottle with a rounded body, narrow neck, and flat base. |
| B03 | Handled mug | A rigid mug with a cylindrical body, one side handle, and a flat base. |
| B04 | Rectangular box | A closed rectangular box with unequal side lengths, flat faces, and slightly rounded edges. |
| B05 | Wide jar | A closed, short, wide cylindrical jar with a screw-top lid and a flat base. |
| B06 | Spherical object | A rigid, nearly spherical object with a small flattened base for stable placement. |
| B07 | Egg-shaped object | A rigid egg-shaped object with one narrower end and a small flattened base for stable placement. |
| B08 | Pear-shaped object | A rigid pear-shaped object with a broad lower body, narrow upper body, and flat base. |
| B09 | Triangular prism | A solid triangular prism with slightly rounded edges and a flat face suitable for stable placement. |
| B10 | Hexagonal prism | A solid hexagonal prism with six flat side faces, slightly rounded edges, and flat ends. |

## Prompt

```
Create a physically plausible counterfactual version of the provided
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
```
