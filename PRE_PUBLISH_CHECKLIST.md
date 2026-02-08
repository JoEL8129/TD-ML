# Pre-Publish Checklist for GitHub

## ✅ Repository Settings (GitHub Web Interface)

- [ ] **Repository Visibility**: Set to Public (or Private if preferred)
- [ ] **Description**: Add a clear, concise description (e.g., "TouchDesigner Machine Learning Toolkit - ML components for TD workflows")
- [ ] **Topics/Tags**: Add relevant topics:
  - `touchdesigner`
  - `machine-learning`
  - `pytorch`
  - `scikit-learn`
  - `python`
  - `ml`
  - `neural-networks`
  - `dimensionality-reduction`
- [ ] **Website**: Add your Discord link if applicable
- [ ] **Default Branch**: Ensure `main` or `master` is set as default

## ✅ Files & Documentation

- [x] **README.md**: Complete and up-to-date
- [x] **LICENSE**: MIT License file present
- [x] **THIRD_PARTY_LICENSES.md**: Created with all dependencies
- [ ] **.gitignore**: Created (excludes sensitive/user-specific files)
- [ ] **requirements.txt**: Up-to-date and matches environment.yml
- [ ] **environment.yml**: Cleaned up (no system-level dependencies)

## ✅ Security & Privacy

- [ ] **No Sensitive Data**: 
  - [ ] No API keys, passwords, or tokens in code
  - [ ] No personal information in files
  - [ ] Check `TDPyEnvManagerContext.yaml` - contains user path (should be in .gitignore)
  - [ ] Review all files for hardcoded paths, usernames, etc.
- [ ] **Git History**: 
  - [ ] If sensitive data was committed, consider using `git-filter-repo` or BFG Repo-Cleaner
  - [ ] Check: `git log --all --full-history -- "*.yaml"` for any sensitive info

## ✅ Code Quality

- [ ] **Code Comments**: Ensure important functions are documented
- [ ] **File Organization**: Structure is clear and logical
- [ ] **No Debug Code**: Remove any debug prints, test code, or temporary files
- [ ] **Backup Folder**: Consider if `Backup/` folder should be excluded (added to .gitignore)

## ✅ Versioning & Releases

- [ ] **Initial Release Tag**: Create a v1.0.0 or v0.1.0 release tag
  ```bash
  git tag -a v0.1.0 -m "Initial release"
  git push origin v0.1.0
  ```
- [ ] **Release Notes**: Prepare release notes for the first release
- [ ] **Changelog**: Consider creating a CHANGELOG.md (optional but recommended)

## ✅ GitHub Features Setup

- [ ] **Issues**: Enable Issues in repository settings
- [ ] **Discussions**: Enable if you want community discussions
- [ ] **Wiki**: Enable if you want to use wiki
- [ ] **Projects**: Set up if you want project management boards
- [ ] **Branch Protection**: Consider protecting main branch (Settings → Branches)

## ✅ Additional Files (Optional but Recommended)

- [ ] **CONTRIBUTING.md**: Guidelines for contributors (if accepting contributions)
- [ ] **CODE_OF_CONDUCT.md**: Community guidelines
- [ ] **CHANGELOG.md**: Track version history
- [ ] **.github/workflows/**: CI/CD workflows (if applicable)

## ✅ Final Checks

- [ ] **Test Installation**: Follow your own README instructions on a clean system
- [ ] **Links Work**: All links in README are valid
- [ ] **Images Load**: All images display correctly
- [ ] **Spelling/Grammar**: Proofread README and documentation
- [ ] **License Compatibility**: Verify all dependencies are compatible (✅ Done - all permissive licenses)

## 🚀 Ready to Publish!

Once all items are checked, you're ready to:
1. Push to GitHub
2. Create your first release
3. Share with the community!

---

**Note**: The `.gitignore` file has been created to exclude:
- User-specific paths (TDPyEnvManagerContext.yaml)
- TouchDesigner project files (*.toe)
- Python cache files
- Backup folders
- Large model/data files

Make sure to commit the `.gitignore` before pushing if you haven't already!
