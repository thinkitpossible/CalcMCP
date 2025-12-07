[build-system]
requires = ["setuptools>=45", "wheel", "setuptools_scm[toml]>=6.2"]
build-backend = "setuptools.build_meta"

[project]
name = "mcp-scientific-calculator"
version = "1.0.0"
description = "Model Context Protocol server for advanced mathematical calculations (SymPy, NumPy, SciPy, pandas)"
readme = "README.md"
requires-python = ">=3.8"
license = {text = "MIT"}
authors = [
    {name = "MCP Calculator Team"}
]
keywords = ["mcp", "model-context-protocol", "calculator", "math", "sympy", "numpy", "scipy", "pandas"]
classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Mathematics",
    "Topic :: Software Development :: Libraries :: Python Modules",
]
dependencies = [
    "sympy>=1.12",
    "numpy>=1.24.0",
    "scipy>=1.10.0",
    "pandas>=2.0.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "black>=23.0",
    "mypy>=1.0",
]

[project.urls]
Homepage = "https://github.com/YOUR_USERNAME/mcp-scientific-calculator"
Documentation = "https://github.com/YOUR_USERNAME/mcp-scientific-calculator#readme"
Repository = "https://github.com/YOUR_USERNAME/mcp-scientific-calculator"
Issues = "https://github.com/YOUR_USERNAME/mcp-scientific-calculator/issues"

[tool.setuptools]
packages = ["calculator"]

[tool.setuptools.package-data]
calculator = ["*.py", "*.json"]
