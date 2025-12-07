# Scientific Calculator MCP 服务器

> 一个生产级别的 Model Context Protocol（MCP）服务器，为 AI 模型提供高级数学计算功能。支持符号数学（SymPy）、数值计算（NumPy/SciPy）、数据分析（pandas）和图像处理。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)

## 📋 目录

- [快速开始](#快速开始)
- [安装指南](#安装指南)
- [配置说明](#配置说明)
- [使用示例](#使用示例)
- [工具参考](#工具参考)
- [常见问题](#常见问题)

## 🚀 快速开始

### 第一步：安装依赖

```bash
pip install sympy numpy scipy pandas
```

### 第二步：配置 Claude Desktop

编辑 Claude Desktop 的配置文件 `claude_desktop_config.json`：

**Windows 用户：**
```json
{
  "mcpServers": {
    "scientific-calculator": {
      "command": "python",
      "args": ["-u", "F:\\AAchengguoofAI\\cuz_caculat\\mcp_server.py"]
    }
  }
}
```

**macOS/Linux 用户：**
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

### 第三步：重启 Claude Desktop

在 Claude 中就能使用计算工具了。

---

## 📦 安装指南

### 方式 1：通过 pip 安装（推荐）

```bash
pip install mcp-scientific-calculator
```

### 方式 2：从源码安装

```bash
git clone https://github.com/thinkitpossible/CalcMCP
cd CalcMCP
pip install -r requirements.txt
```

### 方式 3：本地开发安装

```bash
git clone https://github.com/thinkitpossible/CalcMCP
cd CalcMCP
pip install -e .
```

---

## ⚙️ 配置说明

### 基础配置

确保在 `claude_desktop_config.json` 中配置了服务器路径：

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

### 环境变量（可选）

```bash
# 设置日志级别（可选）
export LOG_LEVEL=DEBUG

# Windows PowerShell
$env:LOG_LEVEL = "DEBUG"
```

---

## 💡 使用示例

### 1. 符号数学运算

**简化表达式：**
```
用户：简化 (x**2 + 2*x + 1) / (x + 1)
工具调用：symbolic_tool
参数：
  operation: "simplify"
  expression: "(x**2 + 2*x + 1) / (x + 1)"

返回：x + 1
```

**求导数：**
```
用户：求 sin(x) + x^3 对 x 的导数
工具调用：symbolic_tool
参数：
  operation: "derivative"
  expression: "sin(x) + x**3"
  variable: "x"

返回：cos(x) + 3*x**2
```

**求积分：**
```
用户：求 2*x 的不定积分
工具调用：symbolic_tool
参数：
  operation: "integral"
  expression: "2*x"
  variable: "x"

返回：x**2
```

**解方程：**
```
用户：解方程 x^2 - 5*x + 6 = 0
工具调用：symbolic_tool
参数：
  operation: "solve_equation"
  equation: "x**2 - 5*x + 6"
  variable: "x"

返回：[2, 3]
```

### 2. 数值计算

**线性方程组求解：**
```
用户：求解线性方程组：2x + y = 5, x + 3y = 7
工具调用：numpy_tool
参数：
  operation: "linear_algebra"
  array_data: [[2, 1], [1, 3]]
  array_data_b: [5, 7]

返回：[2.0, 1.0]
```

**矩阵运算：**
```
用户：计算矩阵 [[1, 2], [3, 4]] 的行列式
工具调用：numpy_tool
参数：
  operation: "determinant"
  array_data: [[1, 2], [3, 4]]

返回：-2.0
```

### 3. 科学计算

**常微分方程求解：**
```
用户：求解 dy/dx = 2*x，初始条件 y(0)=0
工具调用：scipy_tool
参数：
  operation: "solve_ode"
  expression: "2*x"
  initial_condition: 0

返回：数值解向量
```

**函数积分：**
```
用户：计算 ∫sin(x)dx 从 0 到 π
工具调用：scipy_tool
参数：
  operation: "integrate"
  expression: "sin(x)"
  limits: "0:pi"

返回：2.0
```

---

## 🛠️ 工具参考

### symbolic_tool（符号数学）

提供以下操作：

| 操作 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `simplify` | expression | str | 简化数学表达式 |
| `expand` | expression | str | 展开表达式 |
| `factor` | expression | str | 因式分解 |
| `derivative` | expression, variable, order | str | 求导数（可指定阶数） |
| `integral` | expression, variable, lower_limit, upper_limit | str | 求积分（定积分或不定积分） |
| `limit` | expression, variable, point | str | 求极限 |
| `solve_equation` | equation, variable | str | 解方程 |
| `matrix_multiply` | matrix1, matrix2 | str | 矩阵乘法 |
| `matrix_inverse` | matrix | str | 矩阵求逆 |

**使用示例：**
```json
{
  "tool": "symbolic_tool",
  "input": {
    "operation": "simplify",
    "expression": "x**2 + 2*x + 1"
  }
}
```

### numpy_tool（数值计算）

提供以下操作：

| 操作 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `linear_algebra` | array_data, array_data_b | list | 求解线性方程组 Ax=b |
| `determinant` | array_data | float | 计算矩阵行列式 |
| `eigenvalues` | array_data | str | 计算特征值 |
| `eigenvectors` | array_data | str | 计算特征向量 |
| `statistics` | array_data | dict | 统计信息（均值、标准差等） |
| `fft` | array_data | list | 快速傅里叶变换 |
| `correlation` | array1, array2 | float | 计算两个数组的相关系数 |
| `pandas_read` | file_path, file_type | str | 读取数据文件 |
| `pandas_describe` | file_path | str | 数据摘要统计 |

**使用示例：**
```json
{
  "tool": "numpy_tool",
  "input": {
    "operation": "linear_algebra",
    "array_data": [[2, 1], [1, 3]],
    "array_data_b": [5, 7]
  }
}
```

### scipy_tool（科学计算）

提供以下操作：

| 操作 | 参数 | 返回 | 说明 |
|------|------|------|------|
| `solve_ode` | expression, initial_condition, t_span | str | 求解常微分方程 |
| `integrate` | expression, limits | float | 数值积分 |
| `optimize` | expression, initial_guess | str | 函数优化 |
| `interpolate` | x_data, y_data, x_new | str | 插值 |
| `fft` | array_data | list | 快速傅里叶变换 |
| `root_find` | expression, initial_guess | float | 求根 |
| `curve_fit` | x_data, y_data | str | 曲线拟合 |
| `special_function` | func_type, value | float | 特殊函数计算 |

**使用示例：**
```json
{
  "tool": "scipy_tool",
  "input": {
    "operation": "integrate",
    "expression": "sin(x)",
    "limits": "0:pi"
  }
}
```

---

## ❓ 常见问题

### Q1：如何添加到 Claude Desktop？

**A：** 
1. 打开 Claude Desktop 配置文件夹
   - **Windows**: `%APPDATA%\Claude\`
   - **macOS**: `~/Library/Application Support/Claude/`
   - **Linux**: `~/.config/Claude/`

2. 编辑 `claude_desktop_config.json`

3. 在 `mcpServers` 字段添加配置

4. 保存并重启 Claude Desktop

### Q2：提示 "Python not found" 怎么办？

**A：** 确保 Python 已正确安装并在系统 PATH 中：
```bash
python --version
# 或
python3 --version
```

如果没有找到，使用完整路径：
```json
{
  "command": "C:\\Python312\\python.exe",
  "args": ["-u", "path/to/mcp_server.py"]
}
```

### Q3：模块导入错误怎样修复？

**A：** 重新安装依赖：
```bash
pip install --upgrade sympy numpy scipy pandas
```

### Q4：如何运行测试？

**A：** 
```bash
python verify_production_readiness.py
```

### Q5：如何调试问题？

**A：** 使用完整路径并添加日志：
```json
{
  "command": "python",
  "args": ["-u", "-u", "path/to/mcp_server.py"]
}
```

---

## 📊 项目结构

```
.
├── calculator.py              # 核心计算模块（3个工具函数）
├── mcp_server.py              # MCP 服务器实现
├── mcp_protocols.py           # MCP 协议定义（可选）
├── requirements.txt           # 依赖列表
├── setup.py                   # 打包配置
├── pyproject.toml             # 项目元数据
├── advanced_math_problems.py  # 测试用例
├── advanced_math_problems.json# 测试数据
├── verify_production_readiness.py  # 验证脚本
├── README.md                  # 英文文档
├── README_ZH.md              # 中文文档（本文件）
├── QUICKSTART.md             # 快速入门
├── DEPLOYMENT.md             # 部署指南
└── .vscode/                   # VS Code 配置
```

---

## 📝 版本信息

- **版本**: 1.0.0（初始版本）
- **发布日期**: 2025 年 12 月 7 日
- **Python 版本**: 3.8+
- **MCP 版本**: 兼容最新版本
- **许可证**: MIT

---

## 🔗 相关链接

- **GitHub**: https://github.com/thinkitpossible/CalcMCP
- **PyPI**: https://pypi.org/project/mcp-scientific-calculator/
- **MCP 协议**: https://modelcontextprotocol.io/
- **SymPy 文档**: https://docs.sympy.org/
- **NumPy 文档**: https://numpy.org/doc/
- **SciPy 文档**: https://docs.scipy.org/

---

## 📧 反馈与支持

如有问题或建议，请提出 Issue 或联系维护者。

**GitHub Issues**: https://github.com/thinkitpossible/CalcMCP/issues

---

**祝您使用愉快！** 🎉
