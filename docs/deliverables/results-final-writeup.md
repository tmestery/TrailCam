# TrailCam Final Results Writeup

**By:** Tyler Mestery

## Summary

All five models were evaluated on the same held-out ENA24 test split (1,474 crops, image-level 70/15/15 split, seed 42). Accuracy and macro-F1 are reported so rare species are not ignored by overall accuracy alone. A fine-tuned ResNet-18 CNN is the strongest model. Classical HSV+LBP models form a solid but weaker baseline. Zero-shot CLIP underperforms every supervised model.

## Test metrics


| Model               | Accuracy | Macro-F1 |
| ------------------- | -------- | -------- |
| Perceptron          | 59.7%    | 0.540    |
| Logistic Regression | 66.1%    | 0.617    |
| SVM                 | 67.8%    | 0.627    |
| CNN (ResNet-18)     | 87.7%    | 0.866    |
| Zero-Shot CLIP      | 50.7%    | 0.425    |


## Comparison

Among classical HSV+LBP models, the RBF SVM performed best (67.8% accuracy, 0.627 macro-F1), with logistic regression close behind and the perceptron weakest (+8.1 points accuracy for SVM over Perceptron).

The CNN (ImageNet-initialized ResNet-18, fine-tuned on ENA24 crops at 224×224 with Adam lr 1e-3, batch size 32, class-balanced loss, and early stopping on validation macro-F1) was the strongest overall model at 87.7% accuracy and 0.866 macro-F1, a large gain over the SVM baseline (+19.9 accuracy / +0.239 macro-F1).

Zero-shot CLIP (ViT-B-32, OpenAI weights, prompts of the form "a camera trap photo of a {species}", no ENA24 fine-tuning) scored 50.7% accuracy and 0.425 macro-F1, below every trained classical model. Without task-specific labels, open-vocabulary matching struggles on night shots, blur, and visually similar species.

## Deployment recommendation

These results support a clear deployment recommendation. A locally runnable CNN trained on labeled camera-trap crops substantially outperforms both engineered-feature classifiers and an off-the-shelf zero-shot vision-language model on this dataset. Zero-shot CLIP is useful when labeled data are unavailable, but it is not competitive here once supervised training is possible. For remote field use without internet or API costs, the fine-tuned CNN is the preferred species classifier; classical models remain cheap baselines but leave a large accuracy gap relative to the CNN.

## Remaining polish

The data pipeline, classical baselines, CNN, and zero-shot CLIP evaluation are complete on the shared splits. Optional follow-ups are deeper error analysis on rare and visually similar species (for example foxes, squirrels, and generic "Bird"), prompt or backbone variants for CLIP, and packaging the trained CNN checkpoint for an offline demo. Residual failure modes that still need human review should be documented alongside the deployment recommendation above.

## How metrics were produced

```bash
python scripts/train.py              # classical
python scripts/train_cnn.py          # CNN
python scripts/eval_zero_shot.py     # CLIP
python scripts/merge_metrics.py      # results/all_metrics.json
python scripts/evaluate.py           # print metrics table
```
