# FTW-Galileo Evals for GFMs

- Clone galileo repo in the GFMs root. https://github.com/nasaharvest/galileo/tree/main
- Get all necessary checkpoints and keep them in appropriate dir following this structure
```
                galileo/
                └── data/
                    └── baseline_models/
                        ├── croma/
                        │   ├── CROMA_base.pt
                        │   └── CROMA_large.pt
                        ├── decur/
                        │   └── vits16_ssl4eo-s12_ms_decur_ep100.pth
                        ├── dofa/
                        │   └── DOFA_ViT_large_e100.pth
                        ├── mmearth/
                        │   ├── mmearth-atto-checkpoint-199.pth
                        │   └── smaller_model.pth
                        ├── prithvi/
                        │   └── prithvi/
                        ├── satlas/
                        │   └── sentinel2_swint_si_ms.pth
                        ├── satmae/
                        │   ├── fmow_finetune.pth
                        │   ├── fmow_pretrain_model.pth
                        │   ├── fmow_pretrain.pth
                        │   └── satmae_pp.pth
                        └── softcon/
                            └── B13_vits14_softcon.pth

```

- For Galileo specific models the structure is like this
```
            galileo/
            └── data/
                └── models/
                    ├── base/
                    │   ├── config.json
                    │   ├── decoder.pt
                    │   ├── encoder.pt
                    │   ├── second_decoder.pt
                    │   └── target_encoder.pt
                    ├── nano/
                    │   ├── config.json
                    │   ├── decoder.pt
                    │   ├── encoder.pt
                    │   ├── second_decoder.pt
                    │   └── target_encoder.pt
                    └── tiny/
                        ├── config.json
                        ├── decoder.pt
                        ├── encoder.pt
                        ├── second_decoder.pt
                        └── target_encoder.pt
```

- Gedeon will share a folder conaining both `models` and `baseline_models` checkpoint. You simply have to copy paste it into the right path 
