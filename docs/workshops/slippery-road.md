# Slippery Road: a directors' workshop

**Date:** 2026-10-24 · **For:** the third rendition (v3) of *Slippery Road* by R.Amber · **Fixed:** the lyrics and the
music. **Open:** the visual treatment, where Amber appears, where she sings, and every annotation.

A simulated design workshop. The three directors are **fictional composites**, each standing for a tradition of music
video making, not real people:

- **Mara Okonkwo**: the handmade tradition (one physical conceit per song, in-camera tricks, objects with feelings).
- **Teo Lindqvist**: the observational tradition (long takes, restraint, the uncanny in the ordinary, the camera as a
  witness that doesn't help).
- **Iris Varga**: the graphic tradition (colour and light as the emotional system, motifs that rhyme across the cut).

They had in front of them: the song, the songwriter's v1 annotations and two `[Singer: ...]` directions, the
songwriter's review of v1 frame by frame, and what the pipeline can and can't do (below).

## What they were told the machine can do

- Keyframes by MOSS (strong on night, wet roads, trains, light; strong styles if asked) and by Qwen-Image-Edit for
  Amber from her one photo (holds her face and hair; poses can go wrong: hands with six fingers, a body cut off at the
  waist, a 45° tilt; it follows the prompt over the photo when the prompt describes someone else).
- Video by LTX-2.5 with a Transition LoRA: one continuous take that morphs each scene into the next. It morphs well
  when told *what turns into what* (the trigger-word discovery, 2026-10-24); otherwise it cuts or cross-fades.
- Lip-sync by InfiniteTalk from a still: convincing, about 2.8 minutes of compute per second, so it should be spent on
  a few moments, not everywhere.
- One still per bar (about 2.7 s at this tempo); near-duplicate bars merge into scenes.

## The songwriter's review, as the brief

What works: night, trains, wet roads ("extremely well"); the broken-glass commuter faces ("exceptional"); the TV
showing people themselves; Amber falling beside the puddle ("truly amazing").

What doesn't: random pulse (cut it); cross-fades and cuts instead of morphs; Amber walking slowly towards the camera
with her hands out, the "prayer" gesture; a 45° body tilt standing in for "slipping"; legless and handless bodies,
six fingers; the key held up to a door with no lock; a slumped "overdose" pose; strips of white plastic in her hair;
the upside-down puddle face that isn't her; dogs running in packs instead of waiting; bottles moving oddly; weeds for
"keep the lights on".

What was asked for: bottles as a crunched, swirling dance in time; snakes of light that come from the train lights and
move like snakes; Amber anxious, glancing around, instead of gesturing; slipping away *from* the camera; a drone
spiral up from her until she's a small figure in Covent Garden; a devil-like version of her for "what are they afraid
of seeing"; the TV image for *them*, then a pull-out to many homes doing the same.

## The conversation

**Teo:** Before style, the reading. Who is she? Verse one, she's on a train watching people who won't look at her.
Verse three, she sees a hand in the storm and says "it's me… the wet concrete is my home". Then the songwriter puts her
outside a London flat with a key. So she *has* a home. That isn't a contradiction, it's the song. She's one bad month
from the pavement and she knows it. The reveal is recognition, not biography.

**Mara:** Then the hand in the storm has to be a real person she passes. And when the face turns up, it's hers.
That's the trick. One woman, two lives, same face.

**Iris:** And we can carry that without a word if light does the work. Everyone in this city is lit cold: halogen,
train strips, TV blue. She's the only warm thing in it. Amber light on Amber. Sodium streetlamps, a porch light, a
lighter. Wherever she is, there's an amber pool. When we see the woman on the pavement in that same amber pool, the
audience knows before the lyric does.

**Teo:** I like it because it's honest to what already worked. The songwriter loved the night, the trains, the wet
roads. Don't throw them away for a gimmick. Photographic, cold, rainy London, with one warm colour.

**Mara:** No style word, then? I'd have argued for something handmade, cardboard trains, but the review is clear:
the realism is landing. Fine. My handmade instinct goes into the objects instead: the bottles. Different bottles,
crushed, dented, labels torn, lifted by the wind into a little ballet, a swirl on the beat. They "remember how to
breathe", so they inflate and pop back into shape as they rise.

**Iris:** Same for the snakes. The train's window lights stretch into streaks, the streaks drip onto the wet road and
become snakes of light that slither, curl, raise their heads. Light is alive in this city; people aren't.

**Teo:** My rule is about the camera. No cuts, and the songwriter asked for exactly that. Every change goes *through*
something: a window, a puddle, a screen, a light. We tell the transition model what turns into what. The camera is a
commuter who won't help: it drifts, it watches, it doesn't rescue her.

**Mara:** Where does she sing? Lip-sync is expensive and the review said the gesturing was the worst thing. I'd have
her sing only where she'd actually say it out loud.

**Teo:** Chorus one, under her breath, on the street, glancing around, anxious, not performing. Chorus two, at the
door. That's the heart. She doesn't go in. She stands under the porch light, key in her hand at her side, and turns to
look back down the street at the woman on the pavement, who is her. She sings that one to her.

**Iris:** And the key problem disappears: no key held up to a door. Her hand rests flat on the door, the key hangs from
her fingers, the lock is in the frame and she doesn't use it.

**Mara:** The bridge. "Embrace the dark and wet, hug the arctic wind." Give her the wind: she stands still with her eyes
closed and lets it hit her, hair streaming. Real hair, no plastic strips. Then "why don't they look at me": commuters
flowing past her on a platform, every face turned away.

**Iris:** "What are they afraid of seeing": the train window beside her. Her reflection in the glass is wrong: darker,
sharper, eyes too bright, a shadow-self, the devil-version the songwriter asked for. It's what they see when they look,
so they don't.

**Teo:** "Maybe themselves": then we turn the camera on them. The broken-glass faces the songwriter called exceptional.
Commuters' reflections fracturing in the carriage window. Keep it.

**Mara:** Outro is about them too. "Keep the TV ghosts alive": a man alone on a sofa, staring at a TV that shows him on
the sofa staring at the TV. Pull out through his window, and every window in the tower block is the same blue
flicker. "Keep the lights on": lights in those windows clicking on one by one. "Keep the wine glasses full": two glasses
filled at a kitchen table in one of the windows, no faces.

**Iris:** And "down the slippery road in the night" is the end: back to her, the only amber light, and the camera
spirals up and away over Covent Garden until she's a single warm point among thousands of cold ones.

**Teo:** "Slipping" earlier, the chorus line "why do I feel like I'm going down": not a tilt. She walks away from the
camera down the wet street and the street seems to lengthen; she gets smaller while the camera stays.

**Mara:** And the dogs: "patient in the shadows". Not running. Sitting. Three black dogs sitting under a railway arch,
eyes catching the light, watching her pass. Waiting.

**Teo:** The hand, the puddle face. The songwriter loved her falling beside the puddle and disliked the upside-down face
that isn't her. So no inverted face. She's on the wet pavement under the amber lamp, and we see her reflection in the
puddle the right way up, from above.

**Iris:** Last thing: the hands. Every shot with a hand says it's attached to an arm we can see, five fingers, palm in a
natural position. And no full-body shots where her legs might vanish: medium or close shots of her, wide shots only
where she's small in the frame.

## The result

**Treatment** (`pieces/slippery-road-ws/inputs/visual-treatment.txt`): central London on a winter night, early evening,
cold blue light, rain, wet roads, trains and stations: photographic and real. One rule: everyone and everything is lit
cold (halogen, train strip lights, TV blue) except Amber, who is always in warm amber light (sodium streetlamps, a porch
light). Light is alive (snakes, flickers); people are not. One continuous take: every change goes through glass, water,
a screen or a light. No pulse.

**Where Amber is:** watching from the train (her point of view, verse 1); absent in verse 2 (the city's own life:
bottles, tar); on the street for chorus 1, singing under her breath; her point of view, then the double on the pavement
in verse 3 (the reveal); the door for chorus 2, singing to her double; the wind and the platform in the bridge, with
her shadow-self in the glass; absent for most of the outro (them, their homes); the final spiral up from her.

**Where she lip-syncs:** chorus 1 (first two lines) and chorus 2 (all of it), about 25 s in all. Everything else is the
LTX chain.

**Scene storyboard** (by lyric; transitions in *italics*):

| Section | Line(s) | Shot | Amber |
|---|---|---|---|
| Intro | (instrumental) | A night train crossing a viaduct in the rain, warm windows, blue city | none |
| Verse 1 | The train slips… | Inside, her POV: rain on the window, the city sliding by | pov |
| | Glowing snakes… | *The window's light streaks stretch, drip onto the road outside and come alive:* snakes of light slithering along the wet road | none |
| | The questioning passengers… | *The snakes rush up into the carriage's strip lights:* passengers in cold light, every face turned away from the camera | pov |
| | Going wherever… | *A passenger's dark window becomes the night outside:* the suburbs rushing past | pov |
| Verse 2 | Empty plastic bottles… / Roll across the tracks… | *Wind hits the tracks:* crushed bottles of every kind lift and swirl in a dance on the beat, reinflating as they rise | none |
| | Giant brushstrokes of black tar… | *The bottles fall into wet black tar:* a road crew's tar spreading over a road at night, glossy, like a giant brush | none |
| | The tar speaks low… | *The tar's surface becomes a dark train window:* her face faint in the reflection, unreadable | part |
| Chorus 1 | Why am I here… (×2) | *Her reflection steps out of the window onto a rainy street:* medium shot, she walks, glancing around, anxious, singing under her breath; amber streetlight on her | **sing** |
| | Why do I feel like I'm going down… | She walks away from the camera down the long wet street, getting smaller | scene |
| | The slippery road… / …in the night | The empty street, her amber pool of light moving away | scene |
| Verse 3 | Where are they going… / To their dark empty homes… | *The street's windows light up:* her POV of lit windows, strangers inside | pov |
| | To their TV ghosts… | A window with blue TV flicker on the walls | pov |
| | They say we can be anything… | *Pull back:* a billboard of a smiling model over the wet street, rain running down it | none |
| | The black dogs scamper, patient… | Three black dogs sitting under a railway arch, eyes catching the light, waiting | none |
| | An empty outreached hand… | A hand reaching out from a doorway, attached to an arm in a dark sleeve, five fingers, rain | part |
| | Home — home — less | The doorway: a figure huddled under a blanket in an amber pool of light | none |
| | And then I see… it's me | *The figure looks up:* it's her face, under the amber lamp | scene |
| | It's my hand | The same hand, now clearly hers, palm up on the wet stone | part |
| | The wet concrete is my home | Seen from above: her sitting on the wet pavement, her reflection in the puddle the right way up | scene |
| Chorus 2 | Why am I here… (whole chorus) | *The puddle becomes her front door's glass:* she stands at the door of a London flat under the amber porch light, one hand flat on the door, the key hanging from her fingers; she turns and looks back down the street; close-up, singing | **sing** |
| Bridge | Embrace the dark and wet / Hug the arctic wind | She stands still, eyes closed, the wind hitting her, real hair streaming; rain | scene |
| | Why don't they look at me | A platform, commuters flowing past her, every face turned away | scene |
| | What are they afraid of seeing | Her reflection in the train window beside her is wrong: darker, sharper, eyes too bright, a shadow-self | part |
| | Maybe themselves (×2) | Commuters' faces fracturing in the carriage window like broken glass | none |
| Outro | They've got troubles… / Just to keep the lights on | A tower block at night: lights in its windows clicking on one by one | none |
| | Keep the TV ghosts alive | A man alone on a sofa staring at a TV that shows him staring at the TV; *pull out through the window:* every window the same blue flicker | none |
| | Keep the wine glasses full (×2) | Two wine glasses being filled at a kitchen table seen through a window, no faces | none |
| | Down the slippery road (×3) | Back to her on the street in her amber light; *the camera spirals up and away over Covent Garden* until she is one warm point among thousands of cold lights | scene |

These go into the pipeline as the v3 lyric annotations (`pieces/slippery-road-ws/inputs/lyrics.txt`), the treatment,
and hand-written transitions for the key changes in its `scenes.json`.
