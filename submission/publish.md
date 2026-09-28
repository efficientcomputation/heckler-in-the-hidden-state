# Publish the repository and submit the paper

The paper uses `https://github.com/efficientcomputation/heckler-in-the-hidden-state` as its project address. This folder has not been initialized or pushed by the preparation scripts. Create and publish that repository before submitting the paper.

1. On GitHub, create an empty repository owned by `efficientcomputation`, named `heckler-in-the-hidden-state`. Make it public so readers can access the code. Do not select an initial README, licence, or `.gitignore`; the folder already supplies its README and `.gitignore`.
2. Run these commands in Terminal:

```bash
cd /Users/sam/code/DiffuLM-Steering/heckler-in-the-hidden-state &&
git init -b main &&
git add . &&
git commit -m "Add paper, research code, results, and documentation" &&
git remote add origin https://github.com/efficientcomputation/heckler-in-the-hidden-state.git &&
git push -u origin main
```

3. Open the GitHub repository and confirm its README, `paper/`, `code/`, `data/`, and `artifacts/paper.pdf` are visible. The folder's `.gitignore` includes the final PDF and arXiv ZIP and excludes local build files.
4. Start the arXiv submission and upload `artifacts/arxiv-source.zip`. The ZIP includes LaTeX sources, figures, and supplementary code/data. Do not also upload the standalone PDF or the entire repository folder.
5. Copy the title, authors, abstract, and comments from `submission/arxiv-fields.md` into the submission form. Review arXiv's generated PDF and complete submission.

The ZIP has been compiled locally. arXiv's own server preview must be checked during submission.
