# Problem 1: Naive Bayes

**a.**

```
total words in positive = 23        total words in negative = 22

P(great|+) = 0.26                   P(great|-) = 0.09
P(amazing|+) = 0.35                 P(amazing|-) = 0.05
P(epic|+) = 0.20                    P(epic|-) = 0.18
P(boring|+) = 0.09                  P(boring|-) = 0.23
P(terrible|+) = 0.04                P(terrible|-) = 0.18
P(disappointing|+) = 0.04           P(disappointing|-) = 0.27
```

$POS = P(+) * P(great|+) * P(amazing|+) * P(terrible|+) * P(disappointing|+) = 0.50 * 0.26 * 0.35 * 0.04 * 0.04 = 0.0000858$

$NEG = P(-) * P(great|-) * P(amazing|-) * P(terrible|-) * P(disappointing|-) = 0.50 * 0.09 * 0.05 * 0.18 * 0.27 = 0.0001094$

$P(w|S = +) = POS/(POS+NEG) = 0.0000858/0.0000858+0.0001094 = log(0.44)$

$P(w|S = -) = NEG/(POS+NEG) = 0.0001094/0.0000858+0.0001094 = log(0.56)$

Since $P(w|S = -)) > P(w|S = +)$, we can assume that the model will apply the _NEGATIVE_ label on S.

**b. add-1 smoothing**

```
total words in positive (add1) = 29        total words in negative (add1) = 28

P(great|+) = 0.24                          P(great|-) = 0.11
P(amazing|+) = 0.31                        P(amazing|-) = 0.07
P(epic|+) = 0.21                           P(epic|-) = 0.18
P(boring|+) = 0.10                         P(boring|-) = 0.21
P(terrible|+) = 0.07                       P(terrible|-) = 0.18
P(disappointing|+) = 0.07                  P(disappointing|-) = 0.25
```

$POS = P(+) * P(great|+) * P(amazing|+) * P(terrible|+) * P(disappointing|+) = 0.50 * 0.24 * 0.31 * 0.07 * 0.07 = 0.0001823$

$NEG = P(-) * P(great|-) * P(amazing|-) * P(terrible|-) * P(disappointing|-) = 0.50 * 0.11 * 0.07 * 0.18 * 0.25 = 0.0001733$

$P(w|S = +) = POS/(POS+NEG) = 0.0001823/0.0001823+0.0001733 = log(0.51)$

$P(w|S = -) = NEG/(POS+NEG) = 0.0001733/0.0001823+0.0001733 = log(0.49)$

Since $P(w|S = -) < P(w|S = +)$, we can assume that the model will apply the _POSITIVE_ label on S, which is different from if we had not applied _add-1 smoothing_.

**c.**

We could have `not_word` as one of the features such as `not_disappointing` and `not_great`. This allows the model to notice that a not was place before an adjective and therefore, treat it as a different feature. In the case of the sentence S given, this should improve our positive probability since it will no longer see `disappointing` as a negative feature, but a positive which should increase our accuracy and improve classification.
