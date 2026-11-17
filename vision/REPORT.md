# Vision model: evaluation report

Patch classifier, ResNet18 (ImageNet weights, layer4 + head fine-tuned), 160px, 15 epochs.
Evaluated on the dataset's `valid` split (images never seen in training).

```
              precision    recall  f1-score   support

      intact       0.94      1.00      0.97        16
      damage       1.00      0.88      0.93         8

    accuracy                           0.96        24
   macro avg       0.97      0.94      0.95        24
weighted avg       0.96      0.96      0.96        24

```

Confusion matrix (rows = true, cols = predicted, order ['intact', 'damage']):

```
[[16  0]
 [ 1  7]]
```

## Held-out `test` split (not used for epoch selection)

```
              precision    recall  f1-score   support

      intact       0.91      1.00      0.95        10
      damage       1.00      0.75      0.86         4

    accuracy                           0.93        14
   macro avg       0.95      0.88      0.90        14
weighted avg       0.94      0.93      0.93        14

```
