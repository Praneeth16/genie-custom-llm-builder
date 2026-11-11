# Licensing

Check the license of every model, code repository, and dataset before it enters the project. Record what you found in the task card. The table below was checked in November 2026. Check again before you rely on it.

## Models and code

| Item | Code | Weights | Use in a customer project |
|---|---|---|---|
| V-JEPA 2 and 2.1 (`facebookresearch/vjepa2`) | MIT | The Hugging Face card says MIT. The README does not state a weight license. | Yes. Download at run time. Do not redistribute the weights. |
| DINOv3 | Meta's DINOv3 license | Same license. Each user must request access on Hugging Face. | Yes for use. Redistribution must carry the same license. |
| LeJEPA reference repository | CC-BY-NC-4.0 | None published | No. Write SIGReg from the paper. |
| `facebookresearch/jepa-wms` | CC-BY-NC-4.0 | CC-BY-NC-4.0 | No. Internal research only. |

## Datasets

| Dataset | License status | Use in a customer project |
|---|---|---|
| `lerobot/pusht`, `lerobot/aloha_sim_insertion_human` | MIT tag on Hugging Face | Yes |
| DROID and its mirrors | The project site and repository state no license | Not until the terms are confirmed |
| MVTec AD | CC BY-NC-SA 4.0 | No |
| Something-Something v2 | Research use only | No |
| SMD, SMAP and MSL, SKAB, C-MAPSS, NAB | The code has a license. The data has none stated. | Not bundled. The user may download them for their own evaluation. |

When no public dataset is clearly licensed, generate synthetic data or use the customer's own data. Say in the report which one was used.
