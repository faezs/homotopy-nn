# GitHub Actions Workflows

## deploy-pages.yml

Automatically deploys the blog post from `docs/` to GitHub Pages.

**Triggers**:
- Push to `main` or `claude/nix-develop-setup-01GUiQF2cgjkUAZLphdjCLdA` branches
- Changes to `docs/**` files
- Manual workflow dispatch

**Setup Required**:

1. Go to repository Settings → Pages
2. Under "Build and deployment":
   - Source: **GitHub Actions** (not "Deploy from a branch")
3. Push this workflow file to trigger deployment

**URL**: https://faezs.github.io/homotopy-nn/

The workflow:
1. Checks out the repository
2. Uploads the `docs/` folder as a Pages artifact
3. Deploys to GitHub Pages environment
