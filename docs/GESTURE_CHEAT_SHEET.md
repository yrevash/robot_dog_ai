# Robot Dog Gesture Guide (Strict)

This guide matches the current default gesture rules in `face_control_center.py`.

## Important Strict Rule
- Gestures are accepted only when palm is confidently facing the camera.
- Back-side hand or strongly rotated hand is treated as `UNKNOWN`.
- Only one hand is processed at a time, and only for the authorized face.

## Start Sequence
1. Start camera.
2. Start recognition.
3. Wait for `AUTHORIZED: <name>`.
4. Show one clear hand sign and hold for about `0.4-0.8 sec`.

## Gesture -> Command
| Command | Hand Sign |
|---|---|
| `forward` | Open palm (all 5 fingers up) |
| `backward` | Thumb down + other fingers closed |
| `left` | Only index up, pointing left in camera view |
| `right` | Only index up, pointing right in camera view |
| `bark` | Thumb-index pinch + middle/ring/pinky up |
| `stand` | Index + middle + ring up, pinky closed |
| `tail_wag` | Only pinky up |
| `walk` | 4 fingers up (index/middle/ring/pinky), thumb folded |
| `sit` | V sign (index + middle up, ring/pinky closed) |
| `stop` | Fist (all closed) |

## Notes
- `greet` is sent automatically when a person becomes authorized.
- Bark sound uses the selected audio file.
- IoT payload includes `person`, `command`, `source`, `timestamp`.
