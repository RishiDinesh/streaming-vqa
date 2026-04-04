# Qualitative Examples

These examples were selected from the promoted subsample runs and illustrate where the methods help, where they fail, and where the local judge is too coarse to fully separate behaviors.

## 1. `duo_streaming` fixes a `full_streaming` over-interpretation
- Dataset / slice: `RVS-Movie subsample5_movie_offset5`
- Comparison: `full_streaming` vs `duo_streaming (s=0.5)`
- Question: `What starts the series of confrontations in the video?`
- Reference: `A state policeman pulls over the women for speeding.`
- `full_streaming` (`0.6`): `The series of confrontations begins with a man in a police uniform, who is seen holding a gun and pointing it at another man who is sitting in a car.`
- `duo_streaming` (`0.8`): `The series of confrontations begins with the man in the police uniform approaching the police officer in the car.`
- Note: Duo stays closer to the police-stop structure instead of escalating immediately into the wrong gun-pointing narrative.

## 2. `duo_streaming` stays closer to visible scene structure
- Dataset / slice: `RVS-Movie subsample5_movie_offset5`
- Comparison: `full_streaming` vs `duo_streaming (s=0.5)`
- Question: `What setting houses the majority of their confrontations?`
- Reference: `A domestic environment.`
- `full_streaming` (`0.6`): `The majority of their confrontations take place in a dimly lit, industrial setting, with a focus on a room with a large, circular light source that emits sparks/INFO`
- `duo_streaming` (`0.8`): `The setting is a dimly lit room with a large, circular window in the background.`
- Note: Duo is still imperfect, but avoids the stronger industrial/sparks hallucination from full attention.

## 3. `rekv` beats A+B when A+B becomes too uncertain
- Dataset / slice: `RVS-Ego subsample5_offset5`
- Comparison: `rekv` vs `duo_plus_rekv (s=0.375)`
- Question: `What location is shown in the video frame?`
- Reference: `It shows a road.`
- `rekv` (`0.8`): `The video frame shows a person walking on a sidewalk next to a road.`
- `duo_plus_rekv` (`0.6`): `The video frame shows a person walking on a paved road with a white line on the side, possibly a sidewalk or a pedestrian path.`
- Note: A+B adds hedgey uncertainty and extra detail where plain ReKV stays cleaner and more decisive.

## 4. `rekv` beats A+B when A+B drifts into generic description
- Dataset / slice: `RVS-Movie subsample5_movie`
- Comparison: `rekv` vs `duo_plus_rekv (s=0.375)`
- Question: `What starts the sequence of events in the tranquil town?`
- Reference: `The discovery of a severed ear among overgrown grass.`
- `rekv` (`0.8`): `The serene town begins with a peaceful scene of a white picket fence, a red car parked in front of a quaint house, and a serene pond reflecting the clear blue sky.`
- `duo_plus_rekv` (`0.6`): `A man in a white shirt and hat walks through the front yard of a white picket house, surrounded by red roses and a white picket fence.`
- Note: Neither answer is truly correct, but ReKV holds a more coherent opening-scene description while A+B drifts further into generic front-yard narration.

## 5. A+B can match `rekv`
- Dataset / slice: `RVS-Movie subsample5_movie_offset5`
- Comparison: `rekv` vs `duo_plus_rekv (s=0.375)`
- Question: `What recurring element suggests the threat of violence?`
- Reference: `Prevalent use of firearms and preemptive actions.`
- `rekv` (`0.8`): `The recurring element suggesting the threat of violence is the presence of a heavily armed individual in the background, who is holding a gun and appears to be in a state of distress.`
- `duo_plus_rekv` (`0.8`): `The recurring element suggesting the threat of violence is the presence of a heavily armed individual in the background, who is holding a gun and appears to be in a state of distress.`
- Note: This is a clean case where the hybrid preserves ReKV behavior on the current judge.

## 6. The local judge is coarse
- Dataset / slice: `RVS-Movie subsample5_movie`
- Comparison: `full_streaming` vs `duo_streaming (s=0.5)`
- Question: `What's a common element in meetings depicted in the video?`
- Reference: `Conversations over meals.`
- `full_streaming` (`0.8`): `A common element in meetings depicted in the video is the use of a camera to capture the moment.`
- `duo_streaming` (`0.8`): `A common element in meetings depicted in the video is the use of a microphone, as seen in the video where a man is speaking into a microphone while seated at a table.`
- Note: Both answers are visibly off from the reference, yet the local judge assigns the same high score. This is why qualitative inspection remains necessary.
