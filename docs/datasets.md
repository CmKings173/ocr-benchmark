# Dataset sourcing and V1 policy

## Recommended public smoke sources

These datasets are useful for validating the benchmark framework, OCR parsing and robustness. They are not substitutes for the customer shipping-label holdout.

| Source | Best use | Notes |
|---|---|---|
| [CORD](https://github.com/clovaai/cord) | Receipt OCR and post-OCR parsing | 1,000 receipt images, OCR boxes and semantic labels; CC BY 4.0. |
| [SROIE](https://rrc.cvc.uab.es/?ch=13) | Receipt OCR and key information extraction | Use the official challenge terms and preserve the original annotations. |
| [FUNSD](https://github.com/crcresearch/FUNSD) | Key/value forms and word boxes | The original dataset is restricted to non-commercial research/education use. |
| [XFUND](https://github.com/doc-analysis/XFUND) | Multilingual/CJK form stress | Seven languages; repository states CC BY-NC-SA 4.0. |
| [HierText](https://github.com/google-research-datasets/hiertext) | Detection/CER/WER stress | 11,639 natural-scene/document images with word, line and paragraph annotations; verify the underlying Open Images terms. |
| [Barcode dataset directory](https://github.com/BenSouchet/barcode-datasets) | Barcode decoder discovery | Includes ParcelBar, InventBar and synthetic barcode datasets; verify the license of each linked dataset. |

Synthetic text generators such as [SynthText](https://github.com/ankush-me/SynthText) and [SynthTIGER](https://github.com/clovaai/synthtiger) are appropriate for controlled blur, rotation, glare and small-text stress. Synthetic scores must not replace customer holdout scores.

## Dataset tiers

```text
public-smoke/       framework and adapter smoke tests
public-stress/      OCR/layout/barcode robustness
synthetic-stress/   controlled degradation and barcode cases
customer-holdout/   production decision set; never tune against it
```

The V1 holdout contract is one physical label per image. Every sample must provide `label_count: 1`, a `label_type`, ground-truth fields and, when present, an exact barcode value. Bbox/multi-label evaluation remains a later phase after region annotations and matching keys exist.

Do not commit customer images or credentials. Public images remain in a separately tracked dataset directory with source URL, version, checksum and license record.
