# Floor-control offline evaluation (gpt-6-luna, 23 motions x 3 reps)

Success = caught: cut by the right listener, after the planted sentence was played, quoting it.
Clean = no cut on a measured, on-topic line. 95% CI: bootstrap over motions.

| Arm | Overclaim caught | Factual error caught | Off-topic caught | All planted | Clean: no cut | Ungrounded cuts |
|---|---|---|---|---|---|---|
| Before: first sentence only, rules | 26% | 26% | 0% | 17% (6%-29%) | 100% | 0/36 |
| Every sentence, rules | 100% | 94% | 0% | 65% (61%-67%) | 100% | 0/134 |
| Every sentence, rules + moderator (new) | 100% | 88% | 93% | 94% (89%-98%) | 100% | 0/198 |
| Every sentence + moderator, LLM only (no rules) | 100% | 88% | 93% | 94% (89%-98%) | 100% | 0/198 |

## Outcome breakdown per planted type

**Before: first sentence only, rules**
- clean: no_cut 100%
- overclaim: missed 74%, caught 26%; mean lag 0.00 sentences
- factual_error: missed 74%, caught 26%; mean lag 0.00 sentences
- off_topic: missed 100%
- detection by planted sentence position: s1 67%, s2 0%, s3 0%, s4 0%

**Every sentence, rules**
- clean: no_cut 100%
- overclaim: caught 100%; mean lag 0.00 sentences
- factual_error: caught 94%, missed 6%; mean lag 0.00 sentences
- off_topic: missed 100%
- detection by planted sentence position: s1 67%, s2 65%, s3 60%, s4 67%

**Every sentence, rules + moderator (new)**
- clean: no_cut 100%
- overclaim: caught 100%; mean lag 0.00 sentences
- factual_error: caught 88%, missed 6%, caught_by_other 6%; mean lag 0.00 sentences
- off_topic: caught 93%, missed 7%; mean lag 0.00 sentences
- detection by planted sentence position: s1 94%, s2 98%, s3 84%, s4 96%

**Every sentence + moderator, LLM only (no rules)**
- clean: no_cut 100%
- overclaim: caught 100%; mean lag 0.00 sentences
- factual_error: caught 88%, missed 6%, caught_by_other 6%; mean lag 0.00 sentences
- off_topic: caught 93%, missed 7%; mean lag 0.00 sentences
- detection by planted sentence position: s1 94%, s2 98%, s3 84%, s4 96%

## Listener requests (all prefixes)

- calls 2208, errors 0, strict parse failures 4
- latency median 1196 ms, p90 1734 ms
- interrupt requests: {'grounded': 445}
- commentator:before_plant: none 55%, queue 45% (n=306)
- commentator:clean: none 64%, queue 36% (n=276)
- commentator:factual_error: interrupt 94%, none 6%, queue 1% (n=170)
- commentator:off_topic: none 41%, queue 59% (n=174)
- commentator:overclaim: interrupt 90%, none 10% (n=174)
- host:before_plant: none 100% (n=306)
- host:clean: none 100% (n=276)
- host:factual_error: interrupt 3%, none 97% (n=174)
- host:off_topic: interrupt 71%, none 29% (n=174)
- host:overclaim: none 100% (n=174)
