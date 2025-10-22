# 项目依赖说明

## 核心依赖（已配置在 pyproject.toml）

根据命令 `pip install pandas pymilvus sentence-transformers tqdm`，以下依赖已添加到 uv 配置：

### 主要依赖包
| 包名 | 版本 | 用途 |
|------|------|------|
| pandas | >=2.3.2 | 数据处理和分析 |
| pymilvus | >=2.6.1 | Milvus 数据库客户端 |
| sentence-transformers | >=3.0.0 | 文本嵌入模型 |
| tqdm | >=4.65.0 | 进度条显示 |

### 自动安装的依赖
sentence-transformers 会自动安装以下依赖：
- torch (PyTorch 深度学习框架)
- transformers (Hugging Face 模型库)
- numpy (数值计算)
- scikit-learn (机器学习工具)
- pillow (图像处理)

## 安装方法

### 使用 uv（推荐）
```bash
# 同步所有依赖
uv sync

# 使用国内镜像加速
uv sync --index-url https://mirrors.aliyun.com/pypi/simple/
```

### 使用 pip
```bash
# 原始命令
pip install pandas pymilvus sentence-transformers tqdm

# 或从 pyproject.toml 安装
pip install -e .
```

## 验证安装

### 快速测试
```bash
# 测试导入
uv run python -c "import pandas, pymilvus, sentence_transformers, tqdm; print('All imports successful!')"

# 运行完整测试
uv run python test_dependencies.py
```

### 版本检查
```bash
# 查看已安装版本
uv run python -c "
import pandas, pymilvus, sentence_transformers, tqdm
print(f'pandas: {pandas.__version__}')
print(f'pymilvus: {pymilvus.__version__}')
print(f'sentence-transformers: {sentence_transformers.__version__}')
print(f'tqdm: {tqdm.__version__}')
"
```

## 使用示例

### 1. 数据处理（pandas）
```python
import pandas as pd

# 读取数据
df = pd.DataFrame({
    'concept_id': ['EVENT_Login_device', 'EVENT_Purchase_amount'],
    'source_type': ['EVENT', 'EVENT'],
    'field_name': ['device', 'amount']
})
```

### 2. 进度条（tqdm）
```python
from tqdm import tqdm
import time

# 显示进度
for i in tqdm(range(100), desc="Processing"):
    time.sleep(0.01)
```

### 3. 文本嵌入（sentence-transformers）
```python
from sentence_transformers import SentenceTransformer

# 加载模型
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# 生成嵌入
sentences = ['用户登录事件', '购买商品事件']
embeddings = model.encode(sentences)
print(f"嵌入维度: {embeddings.shape}")
```

### 4. Milvus 操作（pymilvus）
```python
from pymilvus import connections, Collection

# 连接 Milvus
connections.connect(host='localhost', port='19530')

# 使用 Collection
collection = Collection("Pampers_metadata")
```

## 注意事项

1. **模型下载**: 首次使用 sentence-transformers 时会自动下载模型，可能需要一些时间
2. **CUDA 支持**: 如果有 GPU，可以安装 CUDA 版本的 PyTorch 以加速计算
3. **内存需求**: sentence-transformers 和 torch 需要较多内存，建议至少 8GB RAM
4. **网络要求**: 下载模型需要访问 Hugging Face，可能需要配置代理

## 环境变量（可选）

如果需要使用代理下载模型：
```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
```

指定模型缓存目录：
```bash
export TRANSFORMERS_CACHE=/path/to/cache
export HF_HOME=/path/to/huggingface/cache
```