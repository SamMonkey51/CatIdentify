# 猫品种分类器 
注：数据集和模型文件因文件大小无法上传，数据集请到下方链接下载后按照下文的文件结构放入，模型文件：
- 百度网盘： https://pan.baidu.com/s/1jH6rR4-g-uVkbGJQP3kQjA 提取码: 0721 
- onedrive: https://cxmhqy-my.sharepoint.com/:u:/g/personal/sam_monkey_0128_ink/EcFGYoPZlfVAlDZ8yEVzOfgBUyRfV11DwMZrjZKSyu83vg?e=tPMMhu
这是一个使用 TensorFlow/Keras 构建的卷积神经网络 (CNN) 模型，用于识别 12 种不同的猫品种。项目还包含一个使用 Gradio 构建的 Web 用户界面，方便用户上传猫的图片并进行实时预测。

## ✨ 功能

*   **品种识别**: 能够识别 12 种常见的猫品种。
*   **模型训练**: 包含完整的模型训练脚本 (`CatBreedClassifier.py`)。
*   **Web 界面**: 提供一个简单易用的 Gradio Web 界面 (`GradioApp.py`)，用于上传图片并查看预测结果。
*   **可扩展性**: 可以轻松扩展以支持更多品种。

## 📂 项目结构

```
CatIdentify/
├── CatBreedClassifier.py   # 用于训练猫品种分类模型的脚本
├── GradioApp.py            # 启动 Gradio Web 界面的应用
├── requirements.txt        # 项目依赖的 Python 库
├── CatsDataset/            # 存放训练图片的目录 (按品种分子目录)
│   ├── Abyssinian/
│   ├── Bengal/
│   └── ...                 # 其他 10 个品种
├── Checkpoint/             # 存放训练过程中保存的模型检查点
├── examples/               # 存放 Gradio 界面示例图片
└── README.md               # 项目说明文档
```

## ⚙️ 安装与设置

1.  **克隆项目**
    ```bash
    git clone <your-repository-url>
    cd CatIdentify
    ```

2.  **创建虚拟环境**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    .venv\Scripts\activate     # Windows
    ```

3.  **安装依赖**
    ```bash
    pip install -r requirements.txt
    ```

## 📊 数据集

模型使用了包含 12 个猫品种的图像数据集进行训练。训练数据应放置在 `CatsDataset/` 目录下，并按品种名称创建子目录。

数据集来源:
*   [Oxford IIIT Cats](https://www.kaggle.com/datasets/imbikramsaha/cat-breeds)

支持的品种:
`Abyssinian`, `Bengal`, `Birman`, `Bombay`, `British Shorthair`, `Egyptian Mau`, `Maine Coon`, `Persian`, `Ragdoll`, `Russian Blue`, `Siamese`, `Sphynx`

## 🧠 模型训练

要重新训练模型，请运行 `CatBreedClassifier.py` 脚本。

```bash
python CatBreedClassifier.py
```

*   脚本会自动从 `CatsDataset/` 目录加载数据。
*   训练过程中，模型的检查点会每 5 个 epoch 保存在 `Checkpoint/` 目录下。
*   训练完成后，最终模型将保存为 `CatClassifier.keras`。

## 🚀 启动应用

要启动 Gradio Web 界面并使用模型进行预测，请运行 `GradioApp.py`。

```bash
python GradioApp.py
```

应用启动后，在浏览器中打开提供的 URL (通常是 `http://127.0.0.1:7860` 。您可以通过界面上传猫的图片，模型将预测其品种，并显示各类别的概率。
