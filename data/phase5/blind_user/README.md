# Your own questions (optional)

Put questions you wrote yourself in `questions.txt` in this folder: one question per line, in plain Chinese, the way you would ask the chatbot. Every line needs a chart, so start it with the four pillars and a gender, then a tab, then the question:

```
庚午 辛巳 壬午 甲辰	male	我的空亡在哪？
丁丑 辛亥 甲子 己巳	female	按扶抑一派，这个八字喜什么？
```

- Write the questions before looking at `src/bazireason/indep_rules.py`, `train_data*.py` or the blind sets, so they stay blind.
- No real person's birth data. The pillars can be made up.
- They are labelled the same way as the other blind questions: the router's intent is checked against the intent you meant, and the rule code gives the answer. Questions the code cannot label (predictions, unsupported topics) count as "should be declined".

Phase 5 ran without this file, so every Phase 5 blind-wording result comes from the model-written set in `../blind/`. A file added later is scored as a separate set; it is never mixed into the sealed results.
