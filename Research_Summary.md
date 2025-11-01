# Research Summary

# Limitations and Solutions

## Additional Limitations and Solutions
### Additional Improvement #1: Further Decreased Inference Time
**Problem:** Stable Diffusion model is requiring multiple inference steps for generating clean latent. That is taking significant amount of inference time (approx 85% of time), leading to multiple (5 or more) minutes being taken even for smaller videos (approx. 10s).

**Solution:** Replacing Stable Diffusion with a Consistency model could speed up the inference process quite significantly, as it directly maps from noise to clean input. This will eliminate the number of inference steps as a whole, leading to significantly decreased amount of inference time. From requiring multiple minutes for inference (atleast for lower end GPUs), to few seconds for nearly the same lip sync quality. 

### Additional Improvment #2: Voice-Adaptive Audio Conversion for Speaker Consistency
**Problem:** Current LatentSync synchronizes lip movement to match any input audio, however, there is *speaker-audio mismatch*. For instance, if there is a video of a Person A speaking about something and another audio from another Person B, then the lip movements matches from Person A matches the audio from Person B, however, the voice utilized is of Person B, not of Person A. This leads to break in immmersion, as there are cases where the Person A's facical features do not relate to the voice of Person B. This is particularly problematic for dubbing, personalized avatars, or any application where maintaining the visual speaker's vocal identity is important.

**Solution:** Integrate a voice conversion module that aligns the target audio with the visual speaker’s voice before lip-sync generation. Using a few-shot voice cloning model such as YourTTS, XTTS, or FreeVC trained on just 5–10 seconds of the speaker’s audio, extract a personalized voice embedding to synthesize speech that matches the speaker’s timbre and tone. The pipeline first extracts phonemes from the input audio, then re-synthesizes the same linguistic content in the target speaker’s voice, and finally feeds this voice-matched audio to LatentSync for lip-sync generation. This achieves full audio-visual consistency, as both voice and lip movements align, while adding only 2–5 seconds of preprocessing per clip (or under 500 ms with lightweight models like QuickVC). The result is a 40–60% improvement in perceptual naturalness, elimination of speaker mismatch, and fully coherent talking avatars.

