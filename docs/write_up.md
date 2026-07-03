# Steering the writing style of an LLM

**This is the write-up for my attempt at ablating the purple prose and euphemisms out of Gemma 4 LLM family.**

### Why?

AI slop gets boring and vomit-inducing after a while. It's pervasive, meaningless, and ruins the reading experience. The most notorious manifestations are purple prose and euphemisms.

Purple prose and euphemisms are not pretrained behaviors. They are intentional post-training artifacts, and they can be safely removed.

### How?

A transformer-based LLM reads text as a vector (a list of numbers) for each word. This vector travels down the dozens of layers that it has. Every layer reads and writes to the same "scratchpad" called a "residual stream", it reads what's there, thinks, then adds its contribution back, and nothing is erased in this process. Gemma4-E4B-it has 42 layers, the scratchpad is a list of 2560 numbers wide.

In our case, every layer in the RL'd model pushes its representation towards purple-ness, accumulating layer by layer. The final result is a very purple list of numbers. This shift is geometrically consistent, they're pushed in the same direction. Our target is to find this direction and remove it from the weights.

**Finding the direction**

We take hundreds of pairs of opposite sentences - written once plainly, and once purply. Then we look at where each of them lands in the model. Then subtract the plain from the purple => The leftover arrow is the purple direction.

This works because each pair shares the same topic so everything except style cancels out.

```direction d = average(purple sentences) − average(plain sentences)```

**Removing it**

A layer does a lot internally, but when it's finally time to write to the residual stream, every layer does it through two matrices: Attention (reader) and MLP (writer) - o_proj and down_proj respectively in our Gemma 4 E4B case. This is exactly where we intervene.

Everything that these matrices write is a vector. We want to guarantee that none of what they add to the scratchpad points along our purple direction `d`. So for each matrix `W` we subtract off the part of its output that lies along `d`:

```W' = W − d (dᵀ W)```

Geometrically this is a projection: we're flattening the matrix against the wall that is perpendicular to `d`. After this edit, no matter what text comes in, whatever `W` writes has zero component along the purple direction. The layer can still say everything it used to - it just can't push the scratchpad toward purple-ness anymore. Do this to every writer in every layer and the accumulation from before has nowhere to build up.

In practice we don't slam it to zero everywhere. Each layer gets a dial `w` in `[0,1]`:

```W' = W − w · d (dᵀ W)```

`w=1` removes the direction completely; `w<1` just turns it down. Early and late layers are doing bookkeeping (reading tokens, choosing the next word) and are fragile to touch, while the middle layers are where the style actually lives. So we ride a soft bump over the depth - gentle at the ends, strongest in the middle - and tune its shape. We also ablate the MLP writer more gently than attention, because zeroing the MLP is more damaging to coherence.

One thing to consider: in a recent dissection (https://www.reddit.com/r/LocalLLaMA/comments/1s1t5ot/rys_ii_repeated_layers_with_qwen35_27b_and_some/), it was discovered that duplicating middle-layers improved the model's "intelligence". So these layers are clearly the most important and need to be handled with care.

Crucially, all of this happens once on the weights themselves. There's no hook and no extra vector added at inference.

But this process is not free lunch. We're essentially delivering a thousand cuts to the model's brain and see what doesn't make it a vegetable afterwards. This is more or less Muntzing but for LLMs instead of electronics (https://en.wikipedia.org/wiki/Muntzing)

### How is this different from finetuning?
**Better:**

- Require much less training data and GPU hours (only needs a few thousand samples).
- Preserve the base model's intent.
- No catastropic forgetting.
- The output is lora-like, the strength can be tuned at serve time.
- If you're finetuning right, all your models will lose all their personalities and converge into what's in your dataset - Gets old after a while!

**Worse:**

- Preserving the base model's intent is a double-edged knife. If the base model omits something, the ablated model will never write it. One example: The base model is RL'd to fade-to-black on explicit scenes => The ablated model will do the same.
- The model cannot go purple conditionally where it'd be appropriate.
- Collateral damage: if the dataset is not careful, vivid will be confused for purple and colaterally removed.

### The journey

I started out with Orb Frontend, its original mission was to break the RP into smaller, more digestible tasks for the writer LLM, then tell an editor model to rewrite any detected slop while keeping the semantics. The editing came with a latency cost. But the process got me some useful tools like **repetition detectors**, which could detect common types of repetition in LLM output - exact phrase, structural, template, openers, these would come in handy later.

But these tools were not enough. Purple slop is varied and can be written in various forms, so exact word detection doesn't cut it. With this limitation in mind, I finetuned a **classifier** to detect AI purple prose. I manually annotated half of the dataset, all in all the process was much faster than I expected. After two weeks I got a classifier with 90% accuracy at detecting common Gemma 4 slop. The original goal was for Orb to use it in combination with the algorithmic detectors above. Ettin was my pick, it's a modern BERT-based text classifier. Though I underuse it by only scoring sentences even though it has an 8k token context window.

This is all too auxiliary and the latency is unacceptable, what if the model didn't write slop at all? What if we removed the ability to write slop? This brings us back to ablation, inspired by https://github.com/p-e-w/heretic

### The pipeline

**Building the dataset**

I mined wiki and news articles and some human writing as good, non-purple examples. Then I prompted Gemma 4 31B to rewrite them into more purple versions. These created synthetic pairs, which would be used for training the classifier and calculating the direction vector.

**Training the classifier**

I finetuned the Ettin 400m model to classify purple prose with 90% accuracy, the inaccuracy mostly fell on false positives (vivid but not AI purple). Then I trained the same Ettin base on euphemisms, first it had to be able to detect what's potentially a sentence that belongs in a NSFW scene and what's not, then it had to detect whether it's euphemistic. This was challenging because sentence-level data lacked the context to determine what's what. So I only settled on best-effort, prioritizing precision instead of blanket-flagging sus ones - just let things through if unsure. This classifier had 97% accuracy as we tip-toe to avoid false positives.

**Ablating the LLM**

My hardware is a humble RTX 3090. So let's start small, the perfect beginner's choice was the slop machine Gemma-4-E4B-it.
I ran 100 trials for over a day. Settled on a random one, eyeballed it. At this point my de-euphemism dataset only had a few dozen very extreme examples. So not only did the ablation change the writing style, the characters also became much ruder, swearing and telling each other to fuck off on first turn (I later attempted a fix by removing the dialogue and only keeping the narration in the ablation process).

The similarity with heretic: We cut and do hundreds of trial-and-error attempts until we get something acceptable.

The difference from heretic: Heretic's optimizes for KLD, a clear objective, less fuzzy, faithful logprobs to the base model, also the opposite of what we want. We want to change the model's voice, its word choices, so KLD is in fact, the antithesis. So what do we use instead of KLD to guarantee the model is not brain-damaged and somewhat faithful to the base? The answer is a system of guardrails that includes:

- Repetition detectors (structural, phrase, openers, template, intra-reply)
- Perplexity on human_writing.txt
- Gen perplexity vs base model text
- Final benchmarks ranking

Let's go into the collapse cases first - the brain-damage cases.

**Collapse cases**

Staccato, repetitive openers - these were the worst offenders - `He turns. He waits. He checks his watch.` scored perfectly on the purple prose ladder, but also read like shit. Also subtle repetition like `The train stopped and she got off the train`. This was where the guardrails came in, they directly judged the output and affected the final score. These bad trials would be severely punished.

**How the optimization works**

We can't just eyeball thousands of edits by hand, we need a number to chase. That number is the classifier's `P(purple)`, and the whole search is built around driving it down without breaking the model.

The optimizer is Optuna running a TPE sampler. Think of it as an educated guesser: it proposes a set of ablation knobs (which layer the direction comes from, how hard to cut, how the cut ramps across the stack, how much of it hits the MLP), we score how good that guess was, and it uses every past score to make its next guess smarter. Lower is better.

The model is loaded exactly once and its weights are snapshotted. Each trial then plays out like this:

1. Restore the pristine weights, then apply the trial's proposed edit. No stacking, every trial starts from the same clean base.
2. Run the edited model through a batch of RP scenarios - the "bait" - that are designed to lure it into purple territory. This produces a pile of generated replies.
3. Feed those replies to the **classifier**. Its averaged `P(purple)` over the rollouts *is* the primary score - the headline number we're pushing toward zero. This is the whole point: the classifier is the judge, and the optimizer is trying to write text the judge no longer flags as slop.
4. Run the guardrails over the same replies (repetition, openers, coherence, length collapse, plain-prose perplexity) and add their penalties on top. These stop the optimizer from cheating its way to a low `P(purple)` by lobotomizing the model.
5. Sum primary + penalties, hand that back to Optuna as the trial's objective.

So a single number, `obj = primary + pen`, captures both "did it get less purple" and "is it still a functioning writer." Optuna keeps the best (lowest) trial, and after the search we re-apply that winning set of knobs to the clean weights and save the model.

A few practicalities. Because a full rollout takes ~5-25 minutes depending on the model size, cheap guards run early: if an edit is already brain-dead on plain prose we skip the rollout entirely, and even mid-rollout we can bail once the partial output is provably doomed no matter how the rest turns out. Every completed trial is also mirrored to a log, so a crash at trial 19 of 40 costs the remaining trials, not the whole day - a resume replays the log to warm-start the sampler and runs only what's left.

A log excerpt of the whole engine at work:

```
[I 2026-07-01 11:01:59,948]   trial 82: obj=+1.125 = primary +0.725 + pen 0.400 [euphemism_floor 0.251 euphemism_opener 0.073 euphemism_intra 0.027 euphemism_audit 0.022 coher 0.021 purple_audit 0.003 purple_intra 0.002]  ppl=0.94x gen_ppl=1.02x
[I 2026-07-01 11:01:59,949] Trial 82 finished with value: 1.1251426238038353 and parameters: {'purple_per_layer': False, 'purple_dir_layer': 23, 'purple_max_weight': 0.635682036705582, 'purple_max_pos': 0.3810898963792737, 'purple_min_weight': 0.038912624289942105, 'purple_min_dist': 0.29644527933693476, 'purple_mlp_scale': 0.505213186881448, 'euphemism_per_layer': False, 'euphemism_dir_layer': 10, 'euphemism_max_weight': 0.3603893627274099, 'euphemism_max_pos': 0.2647177352375682, 'euphemism_min_weight': 0.24816103066288336, 'euphemism_min_dist': 0.5710807097888057, 'euphemism_mlp_scale': 0.6828057295660287}. Best is trial 70 with value: 0.6951598026456997.
```

What each field on the trial line means, Optuna minimizes, so lower is better throughout:

- **`obj=+1.125`** - the trial's final score, the number the optimizer ranks on. It's just `primary + pen`.
- **`primary +0.725`** - the headline style scalar we're driving down: the classifier's `P(purple)`/`P(euphemism)` on this trial's rollouts, weighted across both axes. Lower = the edited model wrote less slop. This is the only term we actually want to shrink; everything in `pen` is a guardrail tax on top.
- **`pen 0.400`** - the total penalty, the sum of every guard below. It's what stops the optimizer from "winning" by lobotomizing the model into short, repetitive, or incoherent text that happens to score low on purpleness.
- The `[...]` bracket is that penalty split by which guard fired, biggest first:
  - **`euphemism_floor 0.251`** - fade-to-black guard (euphemism axis only). Punishes the edit for making an intimate scene less explicit than the base model - i.e. dodging into vagueness instead of getting more direct.
  - **`euphemism_opener 0.073` / (`purple_opener`)** - opener collapse: the same sentence-opening word repeated across a reply (`He turns. He waits. He checks...`).
  - **`euphemism_intra 0.027` / `purple_intra 0.002`** - within-reply word/phrase spam and lexical fixation the edit introduced (`The train stopped and she got off the train`).
  - **`euphemism_audit 0.022` / `purple_audit 0.003`** - Orb's cross-message repetition detectors (template / structural / phrase) counting the repetition the edit added on top of the base model's natural rate.
  - **`collapse`** (not shown on this line - it scored 0) - length-collapse guard. Measures how much less the edited model wrote than the base. Model may cheat by clipping the output or not writing.
  - **`coher 0.021`** - brain-damage guard. After the rollout we restore the original weights and score the edited model's own output under the base model - coherent prose scores low, looping/non-sequitur garbage scores high. This catches damage the purpleness classifier is blind to.
  - Each guard is base-anchored: it only charges for repetition/incoherence the edit adds beyond what the unedited model already did, so natural reuse isn't punished.
- **`ppl=0.94x`** - plain-prose capability check: the edited model's perplexity on held-out human writing vs. the base model. `0.94x` = slightly lower than base (fine, no penalty). Well above `1.0x` would mean the edit hurt the model's general reading ability.
- **`gen_ppl=1.02x`** - the coherence ratio behind the `coher` term: how perplexing the edit's own generations are under base weights, relative to base generating its own.

Various optimizations like early-dropout and batching were implemented to speed up the ablation process. The ideal number of trials is in the hundreds, it takes a day to generate 100 trials for the 31B on an RTX PRO 6000. I stopped at 120 because I ran out of credit.

**Choosing the best one**
Despite all this, the final scores didn't tell the whole story. Subtle brain damage was still a huge concern because we couldn't see exactly what had been affected in the LLM black box during the brain surgery.

The final guarantee was benchmarks. We chose the lowest score, derived an acceptable threshold, and tested all trials that scored lower. The best one then would be chosen. This was the closest we had to capabilities retention measurement.

Chosen one: babi benchmark. It's relevant for RP because it's about spatial/state tracking. This is one of the simple questions in the test set:
```
Mary moved to the office.
John went to the kitchen.
Mary moved to the garden.
John went to the bedroom.
Question: Where is Mary?
Answer: garden
```

We compare the benchmark result against the base model's performance. There are also other evals like gsm8k or ifeval, etc. already implemented but we choose babi as the default.

Now the best E4B attempt was good enough from my testing. Let's go bigger.

## Going bigger

Finally, I rented an RTX 6000 PRO node on Vast for $1 an hour to ablate Gemma 4 31B. Here's the interesting part: Just to see what would happen, I did not teach the optimizer what's slop in the direction pairs. It had zero knowledge of slop like "The air is charged with unspoken tension", I simply let the classifiers optimize them away in an unsupervised manner. The results were funky as expected, some of the high-confidence sentences like "Her voice dripped with surgary venom" weren't ablated away. But that's fine. The final model wrote more tersely and with 60% less slop, goal achieved, right?

De-euphemism also worked partially. My regret was that I clamped it down to 0.5 strength because of what I saw when eyeballing the E4B, which in hindsight, was too weak. It was strong enough to banish the usual slop like "arousal", but was not strong enough to push into vulgarity despite the explicit dataset.

The run was badly configured, the data was confounding, but it was a good learning experience. And also, the 31B is a lot less ablation-friendly than the E4B.

**Lessons learned**

The direction wasn't pure style like I'd thought.

Back in "Going bigger" I raised a length-confounding problem: my purple rewrites were always longer than their plain twins. That wasn't as harmless as I had assumed, it leaked straight into the direction vector. Remember `d = average(purple) − average(plain)` - if "purple" in the pairs also means "longer", `d` partly points along "longer sentence". Ablating that d didn't only de-purple, it also collaterally shortened and flattened everything.
I went looking for whether this was actually happening instead of just theorizing. Per layer, I measured how much of `d` points at the plain-class mean (is it partly "generic direction away from plain text"?) and how strongly a sentence's word count predicts its projection onto `d` (is `d` secretly a length detector?). Both purple and euphemism lit up on the second test - so the raw activation geometry alone can't tell you which axis has the real problem.

**Length projection fix**

What separates them is upstream, in the data: purple's pairs run ~10 words longer on the purple side on average. So purple's `d` really is contaminated by length. The vector `d` is now length + purpleness, not just purpleness like initially thought.

The fix is a projection: measure the empirical direction activations move along as word count increases, then subtract that component out of `d` before saving it (`DEPURPLE_PROJECT=length`). What's left should be "ornamental" with the "longer" confound removed.

```
wcc  = (wc - wc.mean())[:, None, None]                      # centered word counts
axis = unit((wcc * (acts - acts.mean(0))).mean(0))          # cov(words, act) = the "longer" axis
r    = raw_r - (raw_r * axis).sum(-1, keepdim=True) * axis  # strip it out of d
```

It worked, sentences became less flat and kept some vividness. Then I tried the same thing for euphemism, which turned out to be a bad idea. The problem is back to the upstream data again. Euphemistic phrases and words are usually very short and bland, my dataset would draw them out to be more creative and explicit, to lengthen the sentences. I intentionally want the model to write longer in this case. This is a dilemma - but an easily resolved one. It's very simple - we remove length confound for purple, but keep it for euphemism.

**Norm-preservation**

The above mentioned asymmetry pointed at a second knob. Norm-preservation restores each write-matrix column to the norm it had before ablation, so it compensates exactly the magnitude the removal cost - and the compensation scales with how much was removed. 

```
ref_norm = W_ref.norm(dim=cdim, keepdim=True)    # norm BEFORE the preserving removal
W.sub_(delta)                                    # remove the direction
W.mul_(ref_norm / W.norm(...).clamp(min=1e-8))   # scale columns back up to ref_norm
```

For a raw euphemism direction the removal is large and strips coherence-supporting magnitude, I suspect this is because of the overly-explicit dataset generated by a different model family (deepseek v4 pro), Gemma 4 would never write those in the first place. Restoring it should bring the prose back (raw+normpres was the coherent-crude winner). For a length-projected purple direction the removal is already gentle and targeted, so re-inflating write magnitude on top over-corrects: the four-way grid test put len+normpres in the worst cell, pushing the model into flat, listicle output (**Assessment:** 1. … **Immediate Plan:** …), it scores low on the purple classifier despite reading nothing like prose. That's a Goodhart trap the scalar alone would never catch. So I keep norm-preservation off wherever length-projection is on.

Both knobs are now decided automatically per axis, straight from the pairs' own word-count delta - but they can be overriden by env vars for a future axis that doesn't fit this pattern.

**Moral of the story**

Upstream data decides _everything_.

## Where to go from here

De-prosing isn't the only use case.

Some ideas: 

- Re-ablate 31B at full strength properly this time, with the new slop examples I recently added.
- Anti-repetition, taking advantage of paragraph-level classifier.
- Better dialogue - classify dialogue with text segmentation and make them more interesting.
- Dialogue slop removal - Slop doesn't only affect the narration; dialogue is also affected.
- Dataset improvements - I didn't spend a lot of time refining the dataset. The synthetic pairs were super terse and that made the model boring. The length-confounding problem (see "The direction wasn't pure style") is patched at the direction level (`DEPURPLE_PROJECT=length`), but the real fix is still length-matching the purple pairs themselves at generation time - just edit the prompt in generate_synthetic.py and regen the pairs. More diverse data is also needed.
- Anti-cordiality - Make the characters less cooperative and agreeable, more hostile even.
- Replace overused slop with the fluff you like.
- Make the model always write from the character's POV with inner thoughts. This may or may not be a straight direction so it may not be possible, just a theory worth testing.
