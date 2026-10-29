# Floor-control offline evaluation (gpt-5.4-nano, 23 motions x 3 reps)

Success = caught: cut by the right listener, after the planted sentence was played, quoting it.
Clean = no cut on a measured, on-topic line. 95% CI: bootstrap over motions.

| Arm | Overclaim caught | Factual error caught | Off-topic caught | All planted | Clean: no cut | Ungrounded cuts |
|---|---|---|---|---|---|---|
| Before: first sentence only, rules | 14% | 20% | 0% | 12% (4%-21%) | 100% | 0/24 |
| Every sentence, rules | 65% | 75% | 0% | 47% (39%-54%) | 91% | 0/111 |
| Every sentence, rules + moderator (new) | 65% | 71% | 62% | 66% (58%-73%) | 91% | 0/151 |
| Every sentence + moderator, LLM only (no rules) | 68% | 71% | 62% | 67% (59%-74%) | 91% | 2/153 |

## Outcome breakdown per planted type

**Before: first sentence only, rules**
- clean: no_cut 100%
- overclaim: missed 86%, caught 14%; mean lag 0.00 sentences
- factual_error: missed 80%, caught 20%; mean lag 0.00 sentences
- off_topic: missed 100%
- detection by planted sentence position: s1 44%, s2 0%, s3 0%, s4 0%

**Every sentence, rules**
- clean: no_cut 91%, false_cut 9%
- overclaim: caught 65%, missed 35%; mean lag 0.33 sentences
- factual_error: caught 75%, missed 22%, wrong_target 1%, premature 1%; mean lag 0.15 sentences
- off_topic: missed 91%, wrong_target 7%, caught_by_other 1%
- detection by planted sentence position: s1 59%, s2 52%, s3 33%, s4 41%

**Every sentence, rules + moderator (new)**
- clean: no_cut 91%, false_cut 9%
- overclaim: caught 65%, missed 35%; mean lag 0.33 sentences
- factual_error: caught 71%, missed 22%, caught_by_other 4%, wrong_target 1%, premature 1%; mean lag 0.16 sentences
- off_topic: caught 62%, missed 33%, wrong_target 4%; mean lag 0.05 sentences
- detection by planted sentence position: s1 67%, s2 76%, s3 64%, s4 57%

**Every sentence + moderator, LLM only (no rules)**
- clean: no_cut 91%, false_cut 9%
- overclaim: caught 68%, missed 32%; mean lag 0.34 sentences
- factual_error: caught 71%, missed 22%, caught_by_other 4%, wrong_target 1%, premature 1%; mean lag 0.16 sentences
- off_topic: caught 62%, missed 33%, wrong_target 4%; mean lag 0.05 sentences
- detection by planted sentence position: s1 70%, s2 76%, s3 64%, s4 57%

## Listener requests (all prefixes)

- calls 2208, errors 0, strict parse failures 34
- latency median 1238 ms, p90 1662 ms
- interrupt requests: {'grounded': 270, 'ungrounded': 2}
- commentator:before_plant: interrupt 0%, none 2%, queue 98% (n=301)
- commentator:clean: interrupt 2%, none 1%, queue 97% (n=269)
- commentator:factual_error: interrupt 62%, none 1%, queue 37% (n=168)
- commentator:off_topic: interrupt 4%, none 7%, queue 89% (n=166)
- commentator:overclaim: interrupt 50%, queue 50% (n=167)
- host:before_plant: none 100% (n=306)
- host:clean: none 100% (n=276)
- host:factual_error: interrupt 2%, none 98% (n=174)
- host:off_topic: interrupt 38%, none 62% (n=173)
- host:overclaim: none 100% (n=174)
