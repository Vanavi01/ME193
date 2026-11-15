# Stop-motion script: "How the Impenetrable Safe sees your fingers"

**Length:** 3:34 (the rendered video, `movie/impenetrable_safe_explainer.mp4`) · **Voice-over:** you · **Frame rate:** 12 frames per second

Part 1 is the **speaking script**, timed to the video (see `movie/TIMING.md`). Each block fits its
scene at a relaxed pace (about 2.5 words per second); sentences marked *(optional)* can be dropped if
you run long. Part 2 is the **in-depth explanation** behind it, not read aloud, so you can answer
questions and write the Notion page.

Every map, weight and percentage in the video comes from the real trained network looking at a real
picture of your hand (`python movie/make_visuals.py` → `movie/visuals/`).

---

# Part 1: Speaking script

## Scene 1: The safe (0:00–0:12)
**On screen:** the guard slides in, the title appears line by line.

> This is the Impenetrable Safe. No keypad, no key: you show a three-digit code with your fingers,
> and a neural network I built from scratch decides what it sees, all on an Arduino UnoQ.

## Scene 2: Press the button (0:12–0:22)
**On screen:** the safe screen, all three LEDs blink, then a hand appears.

> Press the button, all three LEDs blink, and the safe starts listening. From now on, every camera
> picture goes through the network.

## Scene 3: From camera to numbers (0:22–0:40)
**On screen:** `01_camera_to_numbers.png` revealed left to right, then "64 x 64 x 3 = 12,288 numbers".

> First the picture is cropped to the inside of the blue tape, padded to a square, and shrunk to 64
> by 64 pixels. A color picture is really three grids of numbers: how much red, how much green and
> how much blue is in each pixel. Three channels, 12,288 numbers. *(optional)* That's all the
> network ever sees.

## Scene 4: The whole network (0:40–0:58)
**On screen:** the poster, a red dot travels through each stage, then the weight count.

> This is FingerNet. Three convolution layers, each followed by pooling. Every layer makes more
> feature maps than the one before: 8, then 16, then 32, while the maps get smaller. They're
> squashed into 896 numbers, and a final layer scores the six answers. Everything it knows is
> 11,414 weights.

## Scene 5: Convolution (0:58–1:22)
**On screen:** the 8 real kernels, then one worked calculation, then the kernel sliding while the
map fills in.

> A kernel, also called a filter, is a small 3-by-3 grid of weights, one grid for each color
> channel: 27 weights plus a bias. It slides across the picture. At every spot it multiplies the
> 27 pixel values under it by its weights, adds them up, adds the bias, and ReLU turns negatives into
> zero. One kernel makes one feature map. Eight kernels, eight maps.

## Scene 5b: Pooling (1:22–1:37)
**On screen:** an 8×8 piece of a real feature map; a red 2×2 window hops across it and the biggest
value drops into a 4×4 grid.

> After every convolution layer comes max pooling. In each 2-by-2 square, only the biggest value
> is kept. The map shrinks to a quarter of its size, and a small shift of the hand barely changes
> it.

## Scene 6: What layer 1 learned (1:37–1:55)
**On screen:** the 8 layer-1 feature maps appear one per second, then the labels.

> These are the 8 feature maps from layer one, one per kernel. Darker blue means a stronger
> response. Nobody told the kernels what to look for. Maps 2, 3 and 8 respond to skin, maps 4 and 5
> to the white paper, and map 1 to the edges of the fingers.

## Scene 7: Going deeper (1:55–2:11)
**On screen:** the 16 layer-2 maps, then the 32 layer-3 maps.

> Each of layer two's 16 kernels looks at all 8 maps from layer one at once, so it can combine
> skin and edges into finger-like shapes. Layer three's 32 kernels combine those again. The maps
> get smaller, but there are more of them: more kinds of patterns.

## Scene 8: Squash and vote (2:11–2:39)
**On screen:** `07_squash_to_896.png`, then `08_decision.png` with the blue/red weights and the
probability bars.

> Next, every map from all three layers is max-pooled down to 4 by 4, and the results are lined
> up: 896 numbers that describe this hand.
>
> The last layer is a vote. Each answer, fist to five, has its own weight for every one of those
> 896 numbers: blue counts for it, red against. Multiply, add, and each answer gets a score.
> Softmax turns the six scores into probabilities that add up to 100%. Here: three fingers, 97%.

## Scene 9: All six answers (2:39–2:49)
**On screen:** one example per count with its probabilities.

> The same weights handle every count, fist through five, each at 97 to 100%.

## Scene 10: How it learned (2:49–3:09)
**On screen:** training pictures drop in, the training facts appear line by line.

> Training ran on my PC with 720 labelled pictures from three people. The weights start out
> random. For each batch of 32 pictures, the loss measures how wrong the answers are, and
> backpropagation adjusts every weight to make it a little less wrong. After sixty passes: 86% per
> picture on my hand, 66 to 81% on strangers.

## Scene 11: From digits to an open door (3:09–3:29), end card (3:29–3:34)
**On screen:** the real safe screen with a demo code: yellow per digit, green, unlocked, button,
locked.

> So the lock never trusts one picture. A digit needs 90% confidence for 10 pictures in a row, with
> a fist in between. Three correct digits: green, and the servo opens the latch. Three wrong codes:
> a 30-second lockout. Press the button, and it locks again.
>
> *(end card)* Eleven thousand four hundred and fourteen numbers, guarding the door.

---

# Part 2: How it actually works (in depth, not read aloud)

## Words used precisely
| Word | Meaning here |
|---|---|
| **Channel** | One grid of numbers covering the whole image. The input has 3 (red, green, blue amounts). After a conv layer, each feature map is a channel for the next layer. |
| **Kernel / filter** | The same thing. A small block of learned weights, 3×3 wide and as deep as the input has channels. |
| **Feature map** | The output of one kernel slid over the whole input: a grid showing where that kernel's pattern was found. |
| **Pooling** | Shrinking a map by keeping only the maximum of each small region. |
| **Weight / parameter** | A learned number. The network has 11,414. |

**The colors in the figures are only for display.** The network only ever sees plain numbers.
The red, green and blue channel images are drawn in grey (bright = high value). The feature maps are
drawn in shades of blue (dark = strong response), and the weights in blue (positive) and red
(negative). These colors make the numbers easy to read and have nothing to do with the picture's
colors.

## Step by step, with shapes
| Step | Output shape | Learned weights |
|---|---|---|
| Input: crop inside the tape, pad to a square, resize, scale to −1…1 | 3 × 64 × 64 | – |
| Conv 1: 8 kernels of 3×3×3, + ReLU | 8 × 64 × 64 | 8 × (27 + 1) = 224 |
| Max-pool 2×2 | 8 × 32 × 32 | – |
| Conv 2: 16 kernels of 3×3×8, + ReLU | 16 × 32 × 32 | 16 × (72 + 1) = 1,168 |
| Max-pool 2×2 | 16 × 16 × 16 | – |
| Conv 3: 32 kernels of 3×3×16, + ReLU | 32 × 16 × 16 | 32 × (144 + 1) = 4,640 |
| Max-pool 2×2 | 32 × 8 × 8 | – |
| Squash each of the 3 pooled outputs to 4×4 (adaptive max-pool) and join them | (8 + 16 + 32) × 16 = 896 | – |
| Fully connected: 6 outputs × 896 inputs | 6 scores | 6 × (896 + 1) = 5,382 |
| Softmax | 6 probabilities summing to 1 | – |

## Convolution
A kernel is a 3×3 window **as deep as its input**.
- **In layer 1**, the input has 3 channels, so each kernel is 3×3×3 = 27 weights. At one position
  it multiplies the 27 numbers under it (3×3 pixels × red, green, blue) by its 27 weights, adds them
  all up, and adds its bias. The result is **one** number.
- **ReLU** keeps positive results and sets negative ones to 0, meaning "my pattern isn't here".
- **Sliding:** the kernel slides one pixel at a time over all 64×64 positions. The image is padded
  with a 1-pixel border so the output keeps the same size. That gives one 64×64 feature map.
- **Shared weights:** the same 27 weights are used at every position. This is what makes
  convolution efficient: a "skin detector" learned once works anywhere in the image.

## Why the number of maps keeps growing
- **Each kernel produces exactly one map.** So the number of maps a layer outputs is simply how
  many kernels it has, which is a design choice (8, 16, 32). It isn't something the network
  decides.
- **It doubles because pooling halves the width and height.** Halving width and height cuts the
  number of positions by 4×. Doubling the maps keeps enough room for information while the cost
  stays similar: layers 2 and 3 each do about 1.18 million multiply-adds.
- **Deeper layers need more kinds of pattern.** Layer 1 only needs a few simple detectors (skin,
  paper, edge). Deeper layers describe combinations, like a finger pointing up here, or a gap
  between two fingers, and there are many more of those.
- **A deeper kernel sees every map from the layer before.** A layer-2 kernel is 3×3×8: it looks at
  all 8 layer-1 maps at once, so it can learn rules like "skin here AND edge to the right". A layer-3
  kernel is 3×3×16.
- **Each layer sees a bigger part of the hand.** After pooling, one pixel stands for a bigger
  patch of the original image. By layer 3, each output covers roughly a 22×22-pixel region of the
  64×64 input, about a finger's length.

## When pooling happens and why
- **When:** max-pooling (2×2, stride 2) comes right after the ReLU of **every** conv layer, three
  times in total. There is a fourth pooling step at the end: adaptive max-pooling squashes the
  32×32, 16×16 and 8×8 outputs to 4×4 each.
- **Why:**
  1. **Less work:** every pool leaves 4× fewer numbers for the next layer.
  2. **Tolerance to small shifts:** the maximum of a 2×2 square stays the same if the feature moves
     by one pixel.
  3. **Bigger view for deeper layers:** each pixel after pooling stands for a larger area.
- **Why squash all three layers:** the final layer gets both fine detail (layer 1 edges) and the
  overall shape (layer 3).

## The decision
The final layer computes 6 scores. Each score is a weighted sum of the 896 numbers plus a bias.
**Softmax** turns them into probabilities: p_i = e^(score_i) / Σ_j e^(score_j).

The lock then applies its own rules on top: a digit counts only if the same answer is ≥ 90% for
10 frames in a row, with a fist between digits. Altogether it's about 3.25 million multiply-adds
per picture: 13.6 ms on the UnoQ, 2.45 ms on the PC.

## Training
1. **Data:** 720 labelled pictures, 120 per class, from 3 people.
2. **Augmentation:** every time a picture is used, it gets a random rotation (±15°), shift, zoom
   and change in brightness and contrast, so the network can't just memorize exact images.
3. **Forward pass:** a batch of 32 pictures goes through the network.
4. **Loss:** cross-entropy (−log of the probability given to the correct answer), weighted so
   every class counts equally.
5. **Backpropagation:** computes, for each of the 11,414 weights, how the loss changes if that
   weight changes (the gradient).
6. **Adam optimizer:** moves every weight a small step against its gradient (learning rate 0.001).
7. **Repeat:** for 60 passes over the data.

**Honest testing:** accuracy is measured on recording sessions or people that were never used for
training. Neighbouring frames from the same session look almost identical, so testing on them
would make the score look better than it really is.

---

## Shot list (if you film LEGO stop-motion to cut in)
| # | Prop idea | Seconds |
|---|---|---|
| 1 | safe, guard minifig, camera rig | 12 |
| 2 | button press, LEDs, web screen | 10 |
| 3 | printed 64×64 hand, three tinted plastic sheets for R, G, B | 18 |
| 4 | printed poster, a minifig carrying a 1×1 tile through it | 18 |
| 5 | 3×3 LEGO plate sliding over the printed hand grid | 24 |
| 5b | 8×8 grid of tile stacks → keep the tallest of each 2×2 → 4×4 board | 15 |
| 6 | the 8 printed maps, uncovered one by one | 18 |
| 7 | plate stacks: 8 thin, 16 taller, 32 tallest | 16 |
| 8 | long row of tiles (896), six minifigs collecting "votes" | 28 |
| 9 | six poses, a LEGO bar for each winner | 10 |
| 10 | pile of pictures, a LEGO gear turning a little per batch | 20 |
| 11 | the real unlock: hand, LEDs, servo, guard | 25 |

**Tips:** lock your phone camera's exposure and focus, and use a tripod, so frames don't flicker or
jump. Record the voice-over first, then fit the shots to it.
