#!/bin/bash
# MCP Scientific Calculator - Quick Deployment Script

echo "================================"
echo "MCP Scientific Calculator v1.0.0"
echo "Deployment Script"
echo "================================"
echo ""

# Step 1: Git setup
echo "[1/5] Git repository setup..."
git add .
git commit -m "Initial release: MCP Scientific Calculator v1.0.0

- 3 consolidated tools (symbolic_tool, numpy_tool, scipy_tool)
- 10 university-level math problems validated
- Full MCP specification compliance
- Support for SymPy, NumPy, SciPy, pandas
- Image processing and data analysis capabilities"

echo ""
echo "✓ Git commit created"
echo ""

# Step 2: GitHub instructions
echo "[2/5] GitHub repository creation"
echo ""
echo "Manual steps required:"
echo "1. Go to https://github.com/new"
echo "2. Repository name: mcp-scientific-calculator"
echo "3. Description: Production-ready MCP server for advanced mathematical calculations"
echo "4. Public repository"
echo "5. Do NOT initialize with README (we have one)"
echo ""
echo "Then run:"
echo "  git remote add origin https://github.com/YOUR_USERNAME/mcp-scientific-calculator.git"
echo "  git branch -M main"
echo "  git push -u origin main"
echo ""
read -p "Press Enter when GitHub repo is created and commands are run..."

# Step 3: PyPI build
echo ""
echo "[3/5] Building PyPI packages..."
pip install build twine
python -m build

echo ""
echo "✓ Built packages:"
ls -lh dist/

# Step 4: PyPI upload instructions
echo ""
echo "[4/5] PyPI publishing"
echo ""
echo "Test on TestPyPI first (recommended):"
echo "  python -m twine upload --repository testpypi dist/*"
echo ""
echo "Then publish to PyPI:"
echo "  python -m twine upload dist/*"
echo ""
echo "PyPI credentials required:"
echo "  - Create account at https://pypi.org/account/register/"
echo "  - Create API token at https://pypi.org/manage/account/token/"
echo ""
read -p "Press Enter when PyPI upload is complete..."

# Step 5: GitHub release
echo ""
echo "[5/5] GitHub release creation"
echo ""
git tag -a v1.0.0 -m "Release v1.0.0: Production-ready MCP calculator"
git push origin v1.0.0

echo ""
echo "✓ Git tag created and pushed"
echo ""
echo "Create GitHub release:"
echo "1. Go to https://github.com/YOUR_USERNAME/mcp-scientific-calculator/releases/new"
echo "2. Choose tag: v1.0.0"
echo "3. Release title: MCP Scientific Calculator v1.0.0"
echo "4. Description: Copy from README.md Features section"
echo "5. Attach dist/*.whl and dist/*.tar.gz files"
echo "6. Publish release"
echo ""

# Final summary
echo "================================"
echo "Deployment Summary"
echo "================================"
echo ""
echo "✓ Git repository committed"
echo "✓ PyPI packages built (dist/)"
echo "✓ Git tag v1.0.0 created"
echo ""
echo "Next steps:"
echo "1. ✓ Push to GitHub"
echo "2. ✓ Upload to PyPI"
echo "3. ✓ Create GitHub release"
echo "4. → Submit to MCP Community"
echo ""
echo "MCP Community submission:"
echo "  - Repository: https://github.com/YOUR_USERNAME/mcp-scientific-calculator"
echo "  - PyPI package: mcp-scientific-calculator"
echo "  - README.md and config will be auto-validated"
echo ""
echo "Installation test:"
echo "  pip install mcp-scientific-calculator"
echo "  # Then add to Claude Desktop config (see README.md)"
echo ""
echo "================================"
echo "Deployment complete!"
echo "================================"
