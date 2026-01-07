# License Report for fsosar Project

## Project License
**BSD 3-Clause License** - Copyright (c) 2025, Istituto Italiano di Tecnologia

---

## Dependencies and Their Licenses

### Core Dependencies (from environment.yaml)

#### 1. **Python**
- **License**: Python Software Foundation License (PSF License)
- **Type**: Permissive, GPL-compatible
- **URL**: https://docs.python.org/3/license.html

#### 2. **PyTorch** (pytorch)
- **License**: BSD 3-Clause License
- **Type**: Permissive
- **URL**: https://github.com/pytorch/pytorch/blob/main/LICENSE
- **Includes**: torchvision (also BSD 3-Clause)

#### 3. **Transformers** (Hugging Face)
- **License**: Apache License 2.0
- **Type**: Permissive
- **URL**: https://github.com/huggingface/transformers/blob/main/LICENSE

#### 4. **scikit-learn**
- **License**: BSD 3-Clause License
- **Type**: Permissive
- **URL**: https://github.com/scikit-learn/scikit-learn/blob/main/COPYING

#### 5. **Weights & Biases (wandb)**
- **License**: MIT License
- **Type**: Permissive
- **URL**: https://github.com/wandb/wandb/blob/main/LICENSE

#### 6. **Matplotlib**
- **License**: Matplotlib License (PSF-based)
- **Type**: Permissive, based on PSF license
- **URL**: https://matplotlib.org/stable/users/project/license.html

#### 7. **OpenCV (opencv)**
- **License**: Apache License 2.0
- **Type**: Permissive
- **URL**: https://opencv.org/license/

#### 8. **einops**
- **License**: MIT License
- **Type**: Permissive
- **URL**: https://github.com/arogozhnikov/einops/blob/master/LICENSE

#### 9. **CLIP** (OpenAI CLIP)
- **License**: MIT License
- **Type**: Permissive
- **URL**: https://github.com/openai/CLIP/blob/main/LICENSE
- **Note**: Installed via git+https://github.com/openai/CLIP.git

#### 10. **imageio**
- **License**: BSD 2-Clause License
- **Type**: Permissive
- **URL**: https://github.com/imageio/imageio/blob/master/LICENSE

### Additional Dependencies Identified from Code

#### 11. **NumPy**
- **License**: BSD 3-Clause License
- **Type**: Permissive
- **URL**: https://numpy.org/doc/stable/license.html

#### 12. **Pillow (PIL)**
- **License**: HPND License (Historical Permission Notice and Disclaimer)
- **Type**: Permissive
- **URL**: https://github.com/python-imaging/Pillow/blob/main/LICENSE

#### 13. **tqdm**
- **License**: MIT License / MPL-2.0 (dual-licensed)
- **Type**: Permissive
- **URL**: https://github.com/tqdm/tqdm/blob/master/LICENCE

#### 14. **Pandas**
- **License**: BSD 3-Clause License
- **Type**: Permissive
- **URL**: https://github.com/pandas-dev/pandas/blob/main/LICENSE

#### 15. **SciPy**
- **License**: BSD 3-Clause License
- **Type**: Permissive
- **URL**: https://github.com/scipy/scipy/blob/main/LICENSE.txt

---

## License Compatibility Summary

### ✅ All Dependencies Are Compatible
All dependencies use **permissive licenses** that are compatible with your project's BSD 3-Clause License:

- **BSD 3-Clause**: PyTorch, torchvision, scikit-learn, NumPy, Pandas, SciPy
- **BSD 2-Clause**: imageio
- **MIT License**: wandb, einops, CLIP, tqdm
- **Apache License 2.0**: Transformers, OpenCV
- **PSF-based**: Python, Matplotlib
- **HPND**: Pillow

### Key Points

1. **No Copyleft Licenses**: None of the dependencies use GPL or other copyleft licenses that would require your project to be GPL-licensed.

2. **Attribution Requirements**: All licenses require that:
   - Copyright notices are retained
   - License text is included when distributing

3. **Patent Grants**: Apache 2.0 (Transformers, OpenCV) provides explicit patent grants, offering additional protection.

4. **Commercial Use**: All licenses permit commercial use without restrictions.

---

## Recommendations

### For Distribution:
1. **Include this report** or a similar NOTICE file with your distributions
2. **Keep the LICENSE file** in your repository
3. **Attribution**: When distributing binaries, consider including a THIRD_PARTY_LICENSES file with full license texts

### For Academic/Research Use:
1. **Citations**: When publishing, cite the papers for:
   - PyTorch
   - Transformers (Hugging Face)
   - CLIP (if using OpenAI CLIP features)
   - Any specific model architectures you're using

### For Compliance:
- Your BSD 3-Clause license is compatible with all dependencies
- No additional licensing changes are required
- Ensure you retain all copyright notices when redistributing

---

## How to Verify Licenses

To check licenses programmatically, you can use:
```bash
# For conda packages
conda list --explicit

# For pip packages (if using pip directly)
pip-licenses

# To install pip-licenses tool
pip install pip-licenses
```

---

## Additional Notes

- **PyTorch**: Includes some dependencies with different licenses (e.g., third-party code), but all are permissive
- **CLIP**: Installed from GitHub, ensure you pull the latest version with up-to-date license information
- **Custom Code**: The videotransforms module appears to be custom or adapted code - verify its origin and license if it was adapted from another project

---

*Last Updated: January 7, 2026*
*Generated for: fsosar project*
