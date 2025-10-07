# 跨境选品与销售数据分析工具

这是一个使用 Streamlit 构建的交互式Web应用，旨在帮助跨境电商卖家通过数据分析来进行智能选品和销售预测。

本工具整合了多种数据分析技术，包括时间序列预测、K-Means聚类、情感分析和多语言翻译，将复杂的后端逻辑封装在一个用户友好的图形界面中。

## ✨ 主要功能

- **📊 销售预测**: 上传您的Amazon销售报告，应用将使用 ARIMA 模型生成一个交互式的未来30天销售额预测面积图。
- **🛍️ 品类表现**: 自动生成交互式条形图，直观对比不同产品类别（如 Kurta, Set, Western Dress）的平均销售额。
- **🔥 热销品聚类**: 通过 K-Means 算法，根据商品的总销售额、总销量和订单数，自动对商品进行聚类，并筛选出“热销商品”列表。
- **💬 客户情感分析**: 上传产品评论数据，应用将：
    - 使用更精准的 VADER 模型进行情感分析。
    - 如果数据中没有星级（rating），将根据情感分数自动估算一个1-5星的评级。
    - 提供一个交互式范围滑块，让您可以自由筛选高分、低分或任意区间的评论。
- **🌍 非遗描述翻译**: 上传 UNESCO 非遗数据，应用可以将其中的英文描述一键翻译成多种目标市场语言（如德语、法语、西班牙语等），助力本地化营销。

## 🚀 如何部署和使用

请按照以下步骤在您自己的电脑上运行这个应用。

### 1. 准备工作

- **安装 Anaconda/Miniconda**: 推荐使用 Conda 来管理 Python 环境。请从 [Anaconda官网](https://www.anaconda.com/products/distribution) 下载并安装。
- **安装 Git**: 用于从 GitHub 下载代码。请从 [Git官网](https://git-scm.com/downloads) 下载并安装。

### 2. 下载项目代码

打开您的终端 (Terminal / CMD)，使用 `git clone` 命令下载本项目：
```bash
git clone https://github.com/UnknowTT16/streamlit-ecommerce-analysis.git
```
下载完成后，进入项目文件夹：
```bash
cd streamlit-ecommerce-analysis
```

### 3. 创建并配置 Conda 虚拟环境

为了避免与您电脑上其他 Python 项目的库冲突，我们将为这个项目创建一个独立的环境。

- **创建环境**: 在终端中运行以下命令，创建一个名为 `my_project_env` 的新环境。
  ```bash
  conda create --name my_project_env python=3.9
  ```
- **激活环境**:
  ```bash
  conda activate my_project_env
  ```
  *(成功后，您会看到命令行提示符前面出现 `(my_project_env)`)*

### 4. 安装项目依赖

本项目的所有依赖库都记录在 `requirements.txt` 文件中。在**已激活**的环境中，运行以下命令来一键安装：
```bash
pip install -r requirements.txt
```

### 5. 准备数据文件

此应用需要您提供以下 CSV 数据文件才能进行分析。**请将这些文件下载并放置在与 `app.py` 相同的项目文件夹中**：

- `Amazon Sale Report.csv`: 您的亚马逊销售数据。
- `ich001.csv`: UNESCO 非物质文化遗产数据。
- `reviews.csv`: （可选）亚马逊产品评论数据。

*(注意：由于数据文件较大，它们并未包含在 GitHub 仓库中，需要您自行准备。)*

### 6. 启动 Streamlit 应用！

一切准备就绪！现在，在您的终端中（确保仍在项目的根目录下，并已激活 `my_project_env` 环境），运行以下命令：

```bash
streamlit run app.py
```

执行该命令后，您的默认浏览器会自动打开一个新的标签页（地址通常是 `http://localhost:8501`），您就可以看到并开始使用这个数据分析工具了！

---

## 🛠️ 技术栈

- **前端框架**: Streamlit
- **数据处理**: Pandas, NumPy
- **时间序列分析**: Statsmodels (ARIMA)
- **机器学习**: Scikit-learn (K-Means)
- **情感分析**: VADER
- **多语言翻译**: Deep-Translator
- **数据可视化**: Plotly, Seaborn, Matplotlib
