# GitHub Actions Auto-Publishing Setup

## 🎯 Overview

This guide shows you how to automatically publish HQDE to PyPI whenever you push a new version tag to GitHub.

---

## 📋 One-Time Setup (5 minutes)

### Step 1: Get PyPI API Token

1. Go to https://pypi.org/manage/account/token/
2. Click "Add API token"
3. Token name: `HQDE GitHub Actions`
4. Scope: Select "Project: hqde" (or "Entire account" if first time)
5. Click "Add token"
6. **COPY THE TOKEN** (starts with `pypi-...`) - you won't see it again!

### Step 2: Add Token to GitHub Secrets

1. Go to your GitHub repository: https://github.com/Prathmesh333/HQDE-PyPI
2. Click **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret**
4. Name: `PYPI_API_TOKEN`
5. Value: Paste the token you copied (starts with `pypi-...`)
6. Click **Add secret**

### Step 3: Verify Workflow File Exists

The workflow file should already be at:
```
.github/workflows/publish-to-pypi.yml
```

If not, create it with the content from the file I just created.

---

## 🚀 How to Publish (2 methods)

### Method 1: Push with Version Tag (Recommended)

```bash
# Make sure all changes are committed
git add .
git commit -m "Release v0.1.5 - Critical accuracy fixes"

# Create and push version tag
git tag v0.1.5
git push origin main
git push origin v0.1.5
```

**What happens:**
1. GitHub detects the `v0.1.5` tag
2. Automatically runs the workflow
3. Builds the package
4. Publishes to PyPI
5. Done! ✅

### Method 2: Manual Trigger

1. Go to https://github.com/Prathmesh333/HQDE-PyPI/actions
2. Click "Publish to PyPI" workflow
3. Click "Run workflow"
4. Select branch (usually `main`)
5. Click "Run workflow"

---

## 📊 Monitoring the Publish

### Watch the Progress

1. Go to https://github.com/Prathmesh333/HQDE-PyPI/actions
2. Click on the latest workflow run
3. Watch the steps execute:
   - ✅ Checkout code
   - ✅ Set up Python
   - ✅ Install dependencies
   - ✅ Build package
   - ✅ Check distribution
   - ✅ Publish to PyPI

### If Successful

You'll see:
```
✅ Publish to PyPI
   View at: https://pypi.org/project/hqde/0.1.5/
```

### If Failed

Common issues:
- **Token invalid**: Re-create PyPI token and update GitHub secret
- **Version already exists**: Bump version in `pyproject.toml`
- **Build failed**: Check `pyproject.toml` syntax

---

## 🔄 Complete Workflow

### For v0.1.5 Release:

```bash
# 1. Ensure you're on main branch
git checkout main
git pull origin main

# 2. Verify version in pyproject.toml is 0.1.5
cat pyproject.toml | grep version

# 3. Commit all changes
git add .
git commit -m "Release v0.1.5 - Critical accuracy fixes

- Enabled weight aggregation (FedAvg)
- Reduced dropout to 0.15 with diversity
- Added learning rate scheduling
- Added ensemble diversity
- Added gradient clipping

Expected improvements:
- CIFAR-10: +16-21% accuracy
- SVHN: +13-16% accuracy
- CIFAR-100: +31-41% accuracy"

# 4. Create and push tag
git tag v0.1.5
git push origin main
git push origin v0.1.5

# 5. Wait 2-3 minutes and check:
# https://github.com/Prathmesh333/HQDE-PyPI/actions
```

### Verify Publication

```bash
# Wait 2-3 minutes after workflow completes
pip install hqde==0.1.5 --upgrade

# Verify
python -c "import hqde; print(hqde.__version__)"
# Should output: 0.1.5
```

---

## 🎓 Understanding the Workflow

### Trigger Conditions

The workflow runs when:
1. **Tag pushed**: Any tag matching `v*.*.*` (e.g., `v0.1.5`, `v1.0.0`)
2. **Manual trigger**: From GitHub Actions tab

### What It Does

```yaml
1. Checkout code from GitHub
2. Set up Python 3.10
3. Install build tools (build, twine)
4. Build package (creates dist/ folder)
5. Check distribution (validates package)
6. Publish to PyPI (using your token)
```

### Security

- Token is stored as GitHub Secret (encrypted)
- Only accessible to workflow runs
- Never exposed in logs
- Can be revoked anytime from PyPI

---

## 🔧 Customization

### Change Python Version

Edit `.github/workflows/publish-to-pypi.yml`:
```yaml
- name: Set up Python
  uses: actions/setup-python@v5
  with:
    python-version: '3.11'  # Change this
```

### Add Tests Before Publishing

Add before the "Build package" step:
```yaml
- name: Run tests
  run: |
    pip install pytest
    pytest tests/
```

### Publish to TestPyPI First

Add before PyPI publish:
```yaml
- name: Publish to TestPyPI
  env:
    TWINE_USERNAME: __token__
    TWINE_PASSWORD: ${{ secrets.TEST_PYPI_API_TOKEN }}
  run: python -m twine upload --repository testpypi dist/*
```

---

## 📝 Version Tagging Best Practices

### Semantic Versioning

- `v0.1.5` - Patch (bug fixes)
- `v0.2.0` - Minor (new features, backward compatible)
- `v1.0.0` - Major (breaking changes)

### Tag Naming

✅ Good:
- `v0.1.5`
- `v1.0.0`
- `v2.3.1`

❌ Bad:
- `0.1.5` (missing 'v')
- `version-0.1.5` (wrong format)
- `release-0.1.5` (wrong format)

### Deleting Tags (if needed)

```bash
# Delete local tag
git tag -d v0.1.5

# Delete remote tag
git push origin :refs/tags/v0.1.5

# Re-create and push
git tag v0.1.5
git push origin v0.1.5
```

---

## 🐛 Troubleshooting

### Issue: Workflow doesn't trigger

**Check:**
1. Tag format is `v*.*.*` (e.g., `v0.1.5`)
2. Tag was pushed: `git push origin v0.1.5`
3. Workflow file exists at `.github/workflows/publish-to-pypi.yml`

**Solution:**
```bash
# Verify tag exists
git tag -l

# Push tag explicitly
git push origin v0.1.5
```

### Issue: "Invalid token" error

**Check:**
1. Token is stored in GitHub Secrets as `PYPI_API_TOKEN`
2. Token starts with `pypi-`
3. Token has correct scope (project or account)

**Solution:**
1. Generate new token on PyPI
2. Update GitHub Secret
3. Re-run workflow

### Issue: "Version already exists"

**Check:**
```bash
# Check current version
cat pyproject.toml | grep version
```

**Solution:**
```bash
# Bump version in pyproject.toml
# Then create new tag
git tag v0.1.6
git push origin v0.1.6
```

### Issue: Build fails

**Check workflow logs:**
1. Go to Actions tab
2. Click failed workflow
3. Expand "Build package" step
4. Read error message

**Common causes:**
- Syntax error in `pyproject.toml`
- Missing dependencies
- Invalid package structure

---

## 🎉 Quick Reference

### Publish New Version

```bash
# 1. Update version in pyproject.toml
# 2. Commit changes
git add .
git commit -m "Release v0.1.5"

# 3. Tag and push
git tag v0.1.5
git push origin main
git push origin v0.1.5

# 4. Monitor at:
# https://github.com/Prathmesh333/HQDE-PyPI/actions

# 5. Verify (wait 2-3 min):
pip install hqde==0.1.5 --upgrade
```

### Check Published Version

```bash
# On PyPI
https://pypi.org/project/hqde/

# Locally
pip show hqde
```

---

## ✅ Advantages of GitHub Actions

1. **Automated**: No manual PyPI upload
2. **Consistent**: Same build process every time
3. **Secure**: Token stored safely in GitHub
4. **Traceable**: Full logs of every publish
5. **Fast**: Publishes in 2-3 minutes
6. **Free**: GitHub Actions is free for public repos

---

## 📞 Support

If you encounter issues:

1. **Check workflow logs**: Actions tab → Click workflow → View logs
2. **Verify token**: Settings → Secrets → Check `PYPI_API_TOKEN` exists
3. **Test locally**: Run `python -m build` to check for errors
4. **Check PyPI**: Verify package name and version don't conflict

---

**Ready to publish! Just push your tag and GitHub will handle the rest! 🚀**
