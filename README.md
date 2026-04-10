# Explainability and Fairness Auditing in Distracted Driver Detection — Grad-CAM Analysis on VGG16

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange) ![Platform](https://img.shields.io/badge/Platform-Google%20Colab-yellow)

This project applies Grad-CAM to a VGG16 model trained on the State Farm Distracted Driver Detection dataset. Rather than building a new classifier, the focus is on using explainability methods to assess whether model decisions are semantically meaningful and whether attention patterns shift across demographic and visual factors such as skin tone, lighting, or accessories.

---

## Research Question

Can Grad-CAM provide actionable insights into how a deep learning model classifies distracted driving behaviors — and whether those decisions are consistent, justifiable, and fair across demographics and visual contexts?

## Key Findings

- VGG16 trained on 10-class driver behavior data achieved 99.13% accuracy after 5 epochs (loss reduced from 0.51 to 0.03)
- Confusion matrix shows strong per-class performance across all 10 distraction categories
- Grad-CAM heatmaps confirm the model attends to semantically relevant regions (e.g., hands, phones, steering wheel) for correctly classified images
- Fairness audit on a manually curated subset (`newimgs10`) reveals that for non-white drivers with lower prediction accuracy, the model tends to attend to background regions or clothing rather than task-relevant behavior (observed in classes c3, c4, c5, c6, c9), suggesting potential shortcut learning

## Dataset

| File | Description | Source |
|------|-------------|--------|
| State Farm Distracted Driver Detection | 10-class labeled driver images | Kaggle (Montoya et al., 2016) |
| `newimgs10.zip` | Manually curated bias audit subset, organized by class and skin tone (`white` / `non-white`) | Custom |

The State Farm dataset is available at https://www.kaggle.com/competitions/state-farm-distracted-driver-detection. Both datasets are loaded from Google Drive in the notebook.

## Classes

| Label | Behavior |
|-------|----------|
| c0 | Normal driving |
| c1 | Texting — right |
| c2 | Talking on phone — right |
| c3 | Texting — left |
| c4 | Talking on phone — left |
| c5 | Operating the radio |
| c6 | Drinking |
| c7 | Reaching behind |
| c8 | Hair and makeup |
| c9 | Talking to passenger |

## Project Structure
```
├── bias.ipynb      # Main analysis notebook (Google Colab)
└── README.md
```

## Setup & Usage
```bash
pip install torch torchvision matplotlib numpy pandas opencv-python scikit-learn seaborn tqdm Pillow
```

Open the notebook in Google Colab. Mount your Google Drive and place `imgs.zip` and `newimgs10.zip` in the root of your Drive before running. Run cells in order.

## Methods

- Data augmentation (random crop, horizontal flip, normalization) applied prior to training
- VGG16 fine-tuned on 80/20 split of labeled data for 5 epochs using SGD with momentum
- Grad-CAM applied to the last convolutional layer (`model.features[-1]`) to generate spatial activation heatmaps
- Qualitative fairness audit: Grad-CAM heatmaps visually compared across `white` and `non-white` subgroups in the manually curated dataset, with per-class accuracy, precision, recall, and F1 computed for each group

## Limitations

- The State Farm dataset lacks explicit demographic labels; the bias audit relies on a small manually curated subset, making results exploratory rather than statistically conclusive
- Grad-CAM highlights where the model focuses but does not provide causal explanations or guarantee fairness in prediction

## References

Montoya et al. (2016). State Farm Distracted Driver Detection. *Kaggle.* https://www.kaggle.com/competitions/state-farm-distracted-driver-detection

Sikander & Anwar (2019). Driver fatigue detection systems: A review. *IEEE Transactions on Intelligent Transportation Systems.*

Song & Zhang (2022). Risky-driving-image recognition based on visual attention mechanism and deep learning. *Sensors.*

Streiffer et al. (2017). DarNet: A deep learning solution for distracted driver detection. *ACM Digital Library.*

## License

This project is licensed under [CC BY-NC-ND 4.0](https://creativecommons.org/licenses/by-nc-nd/4.0/). You may view and share it with attribution, but may not modify or use it for commercial or academic submission purposes.
