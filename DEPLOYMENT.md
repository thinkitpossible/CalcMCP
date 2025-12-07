# MCP Scientific Calculator - Deployment Guide

## Pre-Deployment Checklist

✅ **Completed:**
- [x] 3 consolidated tools tested (symbolic_tool, numpy_tool, scipy_tool)
- [x] 10 advanced math problems validated
- [x] MCP specification compliance verified
- [x] README.md with service introduction and configuration
- [x] LICENSE file (MIT)
- [x] setup.py for PyPI packaging

## Deployment Steps

### 1. GitHub Repository Setup

**Create a new repository:**
```bash
# Initialize git (if not already done)
cd F:\AAchengguoofAI\cuz_caculat
git init

# Add all files
git add .
git commit -m "Initial release: MCP Scientific Calculator v1.0.0"

# Create GitHub repo and push
# Go to https://github.com/new
# Repository name: mcp-scientific-calculator
# Public repository
# Then:
git remote add origin https://github.com/YOUR_USERNAME/mcp-scientific-calculator.git
git branch -M main
git push -u origin main
```

**Repository structure (current):**
```
mcp-scientific-calculator/
├── README.md                      # ✅ Service introduction (required)
├── LICENSE                        # ✅ MIT License
├── setup.py                       # ✅ PyPI packaging
├── pyproject.toml                 # ✅ Python project metadata
├── requirements.txt               # ✅ Dependencies
├── calculator.py                  # Core calculation library
├── mcp_server.py                  # MCP server implementation
├── advanced_math_problems.py      # Problem set
├── advanced_math_problems.json    # Problem data
├── verify_production_readiness.py # Testing script
└── .gitignore                     # Git ignore rules
```

### 2. PyPI Publishing (Required by MCP Community)

**Prepare for PyPI:**
```bash
# Install build tools
pip install build twine

# Build distribution packages
python -m build

# This creates:
# dist/mcp-scientific-calculator-1.0.0.tar.gz
# dist/mcp_scientific_calculator-1.0.0-py3-none-any.whl
```

**Test with TestPyPI first:**
```bash
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*

# Test installation
pip install --index-url https://test.pypi.org/simple/ mcp-scientific-calculator
```

**Publish to PyPI:**
```bash
# Create PyPI account at https://pypi.org/account/register/
# Create API token at https://pypi.org/manage/account/token/

# Upload to PyPI
python -m twine upload dist/*

# Installation command (for users):
pip install mcp-scientific-calculator
```

### 3. MCP Community Submission

**Required information (auto-extracted from README.md):**

1. **Service Introduction:** ✅ First paragraph of README.md
   > "A production-ready Model Context Protocol (MCP) server providing advanced mathematical calculation capabilities for AI models. Supports symbolic math (SymPy), numerical computing (NumPy/SciPy), data analysis (pandas), and image processing."

2. **Service Configuration:** ✅ Server config section in README.md
   ```json
   {
     "mcpServers": {
       "scientific-calculator": {
         "command": "python",
         "args": ["-u", "path/to/mcp_server.py"],
         "env": {}
       }
     }
   }
   ```

3. **Environment Variables:** ✅ Empty env object (no env vars required)

4. **Type/Category:** Will be auto-detected as "Mathematics" or "Developer Tools"

5. **Service Description:** AI will summarize README.md content

**Submit to MCP Community:**
- Go to MCP community submission portal
- Provide GitHub repository URL: `https://github.com/YOUR_USERNAME/mcp-scientific-calculator`
- Provide PyPI package name: `mcp-scientific-calculator`
- System will auto-validate README.md and extract configuration
- Review and submit

### 4. Post-Deployment

**Create GitHub Release:**
```bash
# Tag the release
git tag -a v1.0.0 -m "Release v1.0.0: Production-ready MCP calculator"
git push origin v1.0.0
```

**On GitHub:**
- Go to Releases → Create new release
- Choose tag: v1.0.0
- Release title: "MCP Scientific Calculator v1.0.0"
- Description: Copy features section from README.md
- Attach dist/*.whl and dist/*.tar.gz files

**Update README badges:**
- PyPI version badge (auto-updates)
- License badge (already included)
- Add GitHub stars/forks badges if desired

## Validation Checklist

Before submitting to MCP community:

- [ ] README.md has clear service introduction (first paragraph)
- [ ] README.md has complete server configuration with JSON example
- [ ] GitHub repository is public
- [ ] PyPI package is published and installable
- [ ] LICENSE file exists
- [ ] All tools tested and working (run `python test_tools.py`)
- [ ] Repository URL and PyPI name match

## Installation Test (End User)

```bash
# Install from PyPI
pip install mcp-scientific-calculator

# Clone repository
git clone https://github.com/YOUR_USERNAME/mcp-scientific-calculator.git
cd mcp-scientific-calculator

# Add to Claude Desktop config
# Edit %APPDATA%\Claude\claude_desktop_config.json (Windows)
# or ~/Library/Application Support/Claude/claude_desktop_config.json (macOS)
{
  "mcpServers": {
    "scientific-calculator": {
      "command": "python",
      "args": ["-u", "/path/to/mcp_server.py"]
    }
  }
}

# Restart Claude Desktop
# Tools will appear: symbolic_tool, numpy_tool, scipy_tool
```

## Support & Documentation

- GitHub Issues: https://github.com/YOUR_USERNAME/mcp-scientific-calculator/issues
- PyPI Page: https://pypi.org/project/mcp-scientific-calculator/
- MCP Specification: https://modelcontextprotocol.io/docs/develop/build-server
