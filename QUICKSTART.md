# Quick Start Guide for MCP Scientific Calculator

## For End Users

### Installation
```bash
pip install sympy numpy scipy pandas
```

### Claude Desktop Configuration

**Windows:**
1. Open `%APPDATA%\Claude\claude_desktop_config.json`
2. Add this configuration:
```json
{
  "mcpServers": {
    "scientific-calculator": {
      "command": "python",
      "args": ["-u", "F:\\path\\to\\mcp_server.py"]
    }
  }
}
```

**macOS/Linux:**
1. Open `~/Library/Application Support/Claude/claude_desktop_config.json`
2. Add this configuration:
```json
{
  "mcpServers": {
    "scientific-calculator": {
      "command": "python3",
      "args": ["-u", "/path/to/mcp_server.py"]
    }
  }
}
```

3. Restart Claude Desktop
4. You'll see 3 new tools: `symbolic_tool`, `numpy_tool`, `scipy_tool`

## Available Tools

### 1. symbolic_tool
Symbolic mathematics operations:
- `simplify`, `expand`, `factor` - algebraic manipulation
- `derivative`, `integral` - calculus
- `solve` - equation solving
- `limit`, `taylor` - advanced calculus
- `matrix` - symbolic matrix operations

**Example:**
```
Calculate the derivative of x³ + 2x²
→ Use symbolic_tool with operation="derivative", expression="x**3 + 2*x**2", variable="x"
```

### 2. numpy_tool
Numerical computing and data analysis:
- `eigenvalues`, `eigenvectors`, `determinant`, `inverse` - linear algebra
- `matmul`, `dot`, `hadamard` - matrix products
- `svd`, `qr`, `cholesky` - matrix decompositions
- `sum`, `mean`, `std`, `max`, `min` - array operations
- `sin`, `cos`, `tan` - trigonometry
- `pandas_describe`, `pandas_corr`, `pandas_value_counts` - data analysis
- `image_stats`, `image_normalize`, `image_threshold` - image processing

**Example:**
```
Find eigenvalues of matrix [[1,2],[2,1]]
→ Use numpy_tool with operation="eigenvalues", matrix_a="[[1,2],[2,1]]"
```

### 3. scipy_tool
Scientific computing:
- `integrate` - numerical integration
- `optimize_minimize`, `optimize_root` - optimization
- `interpolate_linear`, `interpolate_cubic` - interpolation
- `solve_ode` - differential equations
- `fft`, `rfft` - signal processing
- `statistics` - statistical analysis

**Example:**
```
Solve ODE: dy/dt = -2y, y(0)=1, from t=0 to t=5
→ Use scipy_tool with operation="solve_ode", expression="lambda t, y: -2*y", initial_conditions="[1]", t_values="[0,1,2,3,4,5]"
```

## Testing

Run the verification script:
```bash
python verify_production_readiness.py
```

Expected output:
```
[OK] 工具数量正确
[OK] 工具名称完整
[OK] symbolic_tool.derivative 工作正常
[OK] symbolic_tool.integral 工作正常
[OK] symbolic_tool.solve 工作正常
[OK] numpy_tool.determinant 工作正常
[OK] 问题集完整
[OK] 所有验证通过 - 生产就绪!
```

## Example Problems

The server includes 10 university-level math problems:
1. Second-order ODE with initial conditions
2. Matrix eigenvalue decomposition
3. Fourier series expansion
4. Lagrange multipliers (constrained optimization)
5. Complex integration (residue theorem)
6. Heat equation (PDE)
7. Surface tangent planes
8. Linear ODE systems
9. Green's theorem (line integrals)
10. Calculus of variations

See `advanced_math_problems.py` for complete solutions.

## Troubleshooting

**Tools not appearing in Claude Desktop:**
- Check config file syntax (valid JSON)
- Verify Python path in "command" field
- Ensure all dependencies installed: `pip install sympy numpy scipy pandas`
- Restart Claude Desktop after config changes

**Import errors:**
- Install missing packages: `pip install -r requirements.txt`
- Verify Python version ≥ 3.8

**Tool execution errors:**
- Check input format (JSON strings for arrays/matrices)
- Review operation name spelling
- Consult README.md for parameter requirements

## Support

- GitHub: https://github.com/YOUR_USERNAME/mcp-scientific-calculator
- Issues: https://github.com/YOUR_USERNAME/mcp-scientific-calculator/issues
- MCP Docs: https://modelcontextprotocol.io/
