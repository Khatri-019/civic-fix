# Demo voice transcripts (backup)

If live audio fails during the demo, paste the transcript into the text box instead.
The matching `.wav` files in this folder were generated with Windows text-to-speech.

## 1. Broken light — `broken_light.wav` + `broken_light.png`
"The street light outside the boys hostel gate has not been working for three days. The whole road is dark at night and students feel unsafe walking back from the library."

## 2. Garbage — `garbage.wav` + `garbage.png`
"The garbage bin near the canteen has been overflowing since Monday. Waste is spilling onto the road, it smells really bad and stray dogs are spreading it around."

## 3. Water leak — `water_leak.wav` + `water_leak.png`
"There is a water leak from the pipe on the second floor corridor of the academic block. Water has been dripping for two days and the floor is wet and slippery."

## Contradiction demo (text + mismatched image)
Text: "A pipe burst and the whole street is flooded, water everywhere for a week."
Image: `broken_light.png` → the app warns "Your photo doesn't match your description" and offers
Re-upload photo / Edit description / Continue with my description. **Continue** re-classifies from the text
(Plumbing / Water) and the complaint ends with a note that the photo could not confirm the details.
This case is pre-cached in `demo_cache/` (type the text exactly as above).
