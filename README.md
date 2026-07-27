# CLIP Typographic Attack
 
Vision-language models like CLIP don't just *see* images — they also read text
inside them. That's a vulnerability: stamp a misleading word onto a photo, and
CLIP will often classify the image as the **word** rather than the **subject**.
A photo of a dog with "cat" printed across it gets labelled *cat*.
 
This project builds a pipeline that **detects** the adversarial text, **removes**
it (three different ways), and **re-classifies** — then measures how well each
removal method restores the correct prediction.
 
---
 
