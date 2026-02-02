# Quick Publish Guide - HQDE v0.1.5

##  Fastest Way: GitHub Actions (Recommended)

### One-Time Setup (5 minutes)

1. **Get PyPI Token:**
   - Go to https://pypi.org/manage/account/token/
   - Create token for "hqde" project
   - Copy token (starts with `pypi-...`)

2. **Add to GitHub:**
   - Go to https://github.com/Prathmesh333/HQDE-PyPI/settings/secrets/actions
   - New secret: `PYPI_API_TOKEN`
   - Paste token
   - Save

### Publish v0.1.5 (30 seconds)

```bash
# Commit all changes
git add .
git commit -m "Release v0.1.5 - Critical accuracy fixes"

# Create and push tag
git tag v0.1.5
git push origin main
git push origin v0.1.5
```

**Done!** GitHub will automatically:
- Build the package
- Publish to PyPI
- Takes 2-3 minutes

**Monitor:** https://github.com/Prathmesh333/HQDE-PyPI/actions

---

##  Alternative: Manual Publish

If you prefer manual control:

### Windows:
```bash
cd HQDE-PyPI
publish.bat
```

### Linux/Mac:
```bash
cd HQDE-PyPI
chmod +x publish.sh
./publish.sh
```

---

##  Verify Publication

```bash
# Wait 2-3 minutes, then:
pip install hqde==0.1.5 --upgrade

# Check version
python -c "import hqde; print(hqde.__version__)"
# Should output: 0.1.5
```

---

##  What to Do Next

1. **Update Kaggle Notebook:**
   ```python
   !pip install hqde==0.1.5 --upgrade
   ```

2. **Run Training** (40 epochs)

3. **Look for Success Indicators:**
   - "Weights aggregated and synchronized" after each epoch
   - Learning rate shown and decreasing
   - Loss decreasing smoothly

---

##  Expected Results

| Dataset | Before | After | Gain |
|---------|--------|-------|------|
| CIFAR-10 | ~59% | ~75-80% | +16-21% |
| SVHN | ~72% | ~85-88% | +13-16% |
| CIFAR-100 | ~14% | ~45-55% | +31-41% |

---

##  Full Documentation

- **GitHub Actions Setup:** `GITHUB_ACTIONS_SETUP.md`
- **Complete Guide:** `PUBLISH_CHECKLIST.md`
- **What Changed:** `HQDE_v0.1.5_Summary.md`

---

**Recommended: Use GitHub Actions - it's faster and automated! **
