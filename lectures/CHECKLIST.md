# Before Believing a Model's Result — a checklist

Each item names a question, the potential failure it catches, and the cheapest way to check.

## 1. The number itself

- [ ] **What would chance be?** Check the class balance before admiring any accuracy.
- [ ] **How big is the test set?** Accuracy from 30 samples has error bars of roughly ±15 points; most model-vs-model differences are smaller. Report the standard deviation and the n next to every number.
- [ ] **Would a trivial baseline do this?** Composition, color histograms, the majority class, yesterday's value. If the simple representation ties the fancy one, believe the tie.

## 2. Leakage — did the test data influence anything?

- [ ] **Any preprocessing fit on all the data?** Scaling, feature selection, imputation, PCA — all of it must learn from training rows only. Use a Pipeline so cross-validation does this for you.
- [ ] **Any decision made by looking at test performance?** Choosing features, models, hyperparameters, or when to stop training against the test set spends it. Keep a final test set that is touched once.
- [ ] **Near-duplicates across the split?** These can include homologous sequences, repeated subjects or stimulus items, resampled images. Run a near-duplicate check on your representations (`check_duplicates`). Don't fully trust official grouping variables; they miss things.

## 3. Independence — is the test data genuinely new?

- [ ] **Group structure respected?** Subjects, animals, plates, batches, scanners, sites, families, time windows: whatever your version of "the same individual" is, split by it (`GroupKFold`), not by row.
- [ ] **Does the gap between random and grouped splits tell a story?** No gap on a curated set is what clean looks like — but only the check proves it. A large gap means the random-split number was recognition, not generalization.

## 4. Confounds — what else does the label correlate with?

- [ ] **Could the model be reading the batch instead of the biology?** If batch predicts label at all, a model will find it. Compare within-batch and across-batch performance.
- [ ] **What is in the image/signal besides the thing of interest?** Rulers, markings, stains, scanner signatures, background. A model this good at reading data reads all of it.

## 5. Scope — where does the claim end?

- [ ] **Distribution shift:** performance holds for data like the training data. A new lab, cohort, instrument, or season is a new claim requiring new evidence.
- [ ] **Learning curve:** still climbing → more data will help; plateaued → it won't, look elsewhere. Either way, the curve is one plot (`learning_curve`).
- [ ] **Contamination:** if your evaluation items existed publicly (UniProt, GenBank, the internet), a pretrained model may have seen them. Check before benchmarking.

## 6. Interpretation — what the model can and cannot tell you

- [ ] **Saliency and attention maps are not explanations.** Treat them as hypotheses at best.
- [ ] **Wanting "why"?** Run designed experiments on the model. Probe for information, ablate inputs, compare controlled variants. Treat the model as an organism you can experiment on.
- [ ] **Raw models have no guardrails.** Deployed assistants like Claude and ChatGPT are systems (retrieval, tool calls, verification) built around models. Open source models you download are (usually) raw statistical objects. Their failure modes are yours to manage.

---

*Suggested reading behind the checklist:*

* *Zech JR et al. (2018). Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs. PLOS Medicine 15(11): e1002683. [doi:10.1371/journal.pmed.1002683](https://doi.org/10.1371/journal.pmed.1002683) — the pneumonia model that learned which hospital.*
* *Winkler JK et al. (2019). Association between surgical skin markings in dermoscopic images and diagnostic performance of a deep learning convolutional neural network for melanoma recognition. JAMA Dermatology 155(10):1135–1141. [doi:10.1001/jamadermatol.2019.1735](https://doi.org/10.1001/jamadermatol.2019.1735) — melanoma classifiers reading surgical skin markings.*
* *Geirhos R et al. (2020). Shortcut learning in deep neural networks. Nature Machine Intelligence 2:665–673. [doi:10.1038/s42256-020-00257-z](https://doi.org/10.1038/s42256-020-00257-z)*
* *Yarkoni T & Westfall J (2017). Choosing prediction over explanation in psychology: lessons from machine learning. Perspectives on Psychological Science 12(6):1100–1122. [doi:10.1177/1745691617693393](https://doi.org/10.1177/1745691617693393)*
