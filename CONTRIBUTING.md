# 贡献指南

感谢您对MV字幕工具集的关注！我们欢迎各种形式的贡献，包括但不限于：

- 报告问题
- 提交功能请求
- 提交代码修复
- 提交新功能
- 改进文档

## 开发环境设置

1. 克隆仓库：

```bash
git clone https://github.com/euler-gaozl/mv-subtitle-tools.git
cd mv-subtitle-tools
```

2. 创建并激活虚拟环境（可选但推荐）：

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. 安装开发依赖：

```bash
pip install -r requirements.txt
```

## 代码风格

我们使用以下代码风格规范：

- 遵循PEP 8规范
- 使用4个空格进行缩进
- 使用有意义的变量名和函数名
- 为所有函数和类添加文档字符串
- 保持代码简洁明了

## 提交流程

1. 创建一个新分支：

```bash
git checkout -b feature/your-feature-name
```

2. 进行更改并提交：

```bash
git add .
git commit -m "描述你的更改"
```

3. 推送到GitHub：

```bash
git push origin feature/your-feature-name
```

4. 创建Pull Request：
   - 前往GitHub仓库页面
   - 点击"Pull Requests"
   - 点击"New Pull Request"
   - 选择你的分支
   - 填写PR描述
   - 提交PR

## 报告问题

如果您发现问题，请使用GitHub Issues进行报告。请包含以下信息：

- 问题的详细描述
- 复现步骤
- 预期行为
- 实际行为
- 环境信息（操作系统、Python版本等）
- 如果可能，提供截图或日志

## 功能请求

如果您有新功能的想法，请使用GitHub Issues提交功能请求。请包含以下信息：

- 功能的详细描述
- 为什么这个功能对项目有价值
- 如果可能，提供实现思路

## 测试

在提交代码之前，请确保：

1. 所有现有测试都能通过
2. 为新功能添加适当的测试
3. 测试覆盖率不会降低

运行测试：

```bash
python -m unittest discover tests
```

## 文档

如果您的更改影响了API或用户体验，请更新相应的文档：

- README.md
- docs/API.md
- docs/使用教程.md

## 许可证

通过贡献代码，您同意您的贡献将在MIT许可证下发布。

## 联系方式

如果您有任何问题，请通过GitHub Issues或发送电子邮件至euler@example.com与我们联系。

感谢您的贡献！
