import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import math
import os

class PositionalEncoding(nn.Module):
    """位置编码模块"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class ImageTransformerClassifier(nn.Module):
    """Transformer图像分类模型"""
    def __init__(self, num_classes=10, d_model=128, nhead=8, num_layers=3):
        super().__init__()
        
        # 图像到序列的投影 (28x28 = 784)
        self.patch_embed = nn.Linear(784, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=0.1,
            activation='relu',
            batch_first=False
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # 展平图像 [batch, 784]
        x = x.view(batch_size, -1)
        
        # 添加序列维度并投影 [batch, 1, d_model]
        x = x.unsqueeze(1)
        x = self.patch_embed(x)
        
        # 位置编码 [seq_len, batch, d_model]
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        
        # Transformer [seq_len, batch, d_model]
        x = self.transformer(x)
        
        # 全局平均池化并分类 [batch, num_classes]
        x = x.mean(dim=0)
        return self.classifier(x)

class TransformerImagePredictor:
    """
    Transformer图像分类预测器
    
    用于加载训练好的模型并进行预测
    """
    
    def __init__(self, model_path='transformer_image_classifier.pth', device='cpu', invert_colors=False):
        """
        初始化预测器
        
        Args:
            model_path: 模型权重文件路径
            device: 设备 ('cpu' 或 'cuda')
            invert_colors: 是否反转颜色（如果图像是白底黑字，设为True）
        """
        self.device = torch.device(device)
        self.model_path = model_path
        self.invert_colors = invert_colors
        
        # 定义图像预处理变换
        if self.invert_colors:
            # 如果图像是白底黑字，需要反转颜色
            self.transform = transforms.Compose([
                transforms.Grayscale(),           # 转换为灰度图
                transforms.Resize((28, 28)),      # 调整大小为28x28
                transforms.ToTensor(),            # 转换为Tensor
                transforms.Lambda(lambda x: 1.0 - x),  # 颜色反转
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
            ])
        else:
            # 标准MNIST预处理（黑底白字）
            self.transform = transforms.Compose([
                transforms.Grayscale(),           # 转换为灰度图
                transforms.Resize((28, 28)),      # 调整大小为28x28
                transforms.ToTensor(),            # 转换为Tensor
                transforms.Normalize((0.1307,), (0.3081,))  # MNIST标准化参数
            ])
        
        # 加载模型
        self.model = self._load_model()
        
        # 类别名称 (MNIST数字0-9)
        self.class_names = [str(i) for i in range(10)]
    
    def _load_model(self):
        """加载模型权重"""
        # 初始化模型结构
        model = ImageTransformerClassifier(
            num_classes=10,
            d_model=128,
            nhead=8,
            num_layers=3
        )
        
        # 加载权重
        if os.path.exists(self.model_path):
            # 注意：如果模型是用PyTorch Lightning保存的，可能需要调整
            try:
                # 尝试直接加载
                model.load_state_dict(torch.load(self.model_path, map_location=self.device))
            except:
                # 如果失败，可能是PyTorch Lightning保存的checkpoint
                checkpoint = torch.load(self.model_path, map_location=self.device)
                if 'state_dict' in checkpoint:
                    # 提取state_dict
                    state_dict = {}
                    for k, v in checkpoint['state_dict'].items():
                        # 移除'model.'前缀（如果是Lightning保存的）
                        if k.startswith('model.'):
                            state_dict[k[6:]] = v
                        else:
                            state_dict[k] = v
                    model.load_state_dict(state_dict)
                else:
                    raise ValueError("无法加载模型权重")
            
            model.to(self.device)
            model.eval()  # 设置为评估模式
            print(f"模型已从 {self.model_path} 加载")
        else:
            raise FileNotFoundError(f"模型文件 {self.model_path} 不存在")
        
        return model
    
    def preprocess_image(self, image):
        """
        预处理图像
        
        Args:
            image: PIL图像、numpy数组或文件路径
            
        Returns:
            预处理后的图像tensor
        """
        # 如果输入是文件路径
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"图像文件 {image} 不存在")
            image = Image.open(image)
        
        # 如果输入是numpy数组
        elif isinstance(image, np.ndarray):
            # 假设是灰度图像
            if image.ndim == 2:
                image = Image.fromarray(image.astype(np.uint8))
            # 假设是RGB图像
            elif image.ndim == 3:
                image = Image.fromarray(image.astype(np.uint8))
            else:
                raise ValueError("不支持的图像格式")
        
        # 如果输入已经是PIL图像，直接使用
        elif not isinstance(image, Image.Image):
            raise TypeError("不支持的图像类型，请提供PIL图像、numpy数组或文件路径")
        
        # 应用预处理变换
        image_tensor = self.transform(image).unsqueeze(0)  # 添加batch维度
        return image_tensor.to(self.device)
    
    def predict(self, image):
        """
        预测单张图像
        
        Args:
            image: PIL图像、numpy数组或文件路径
            
        Returns:
            dict: 包含预测结果的字典
        """
        # 预处理图像
        image_tensor = self.preprocess_image(image)
        
        # 预测
        with torch.no_grad():
            logits = self.model(image_tensor)
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(logits, dim=1).item()
            confidence = probabilities[0][predicted_class].item()
        
        # 获取所有类别的概率
        all_probabilities = {
            self.class_names[i]: float(prob) 
            for i, prob in enumerate(probabilities[0].cpu().numpy())
        }
        
        return {
            'predicted_class': predicted_class,
            'predicted_label': self.class_names[predicted_class],
            'confidence': confidence,
            'all_probabilities': all_probabilities,
            'top_predictions': sorted(
                [(self.class_names[i], float(prob)) 
                 for i, prob in enumerate(probabilities[0].cpu().numpy())],
                key=lambda x: x[1],
                reverse=True
            )[:3]  # 只返回前3个最可能的预测
        }
    
    def predict_batch(self, image_list):
        """
        批量预测多张图像
        
        Args:
            image_list: 图像列表，每个元素可以是PIL图像、numpy数组或文件路径
            
        Returns:
            list: 预测结果列表
        """
        results = []
        for image in image_list:
            try:
                result = self.predict(image)
                results.append(result)
            except Exception as e:
                results.append({
                    'error': str(e),
                    'predicted_class': None,
                    'predicted_label': None,
                    'confidence': 0.0
                })
        
        return results
    
    def analyze_prediction(self, prediction_result):
        """
        分析预测结果并返回可读的字符串
        
        Args:
            prediction_result: predict方法返回的字典
            
        Returns:
            str: 可读的分析结果
        """
        if 'error' in prediction_result:
            return f"预测错误: {prediction_result['error']}"
        
        result_str = f"预测结果: 数字 {prediction_result['predicted_label']}\n"
        result_str += f"置信度: {prediction_result['confidence']:.4f}\n"
        result_str += "所有类别概率:\n"
        
        for class_name, prob in prediction_result['all_probabilities'].items():
            result_str += f"  {class_name}: {prob:.4f}\n"
        
        result_str += "最可能的三个预测:\n"
        for i, (class_name, prob) in enumerate(prediction_result['top_predictions']):
            result_str += f"  {i+1}. {class_name}: {prob:.4f}\n"
        
        return result_str

# 使用示例和演示代码
if __name__ == "__main__":
    # 创建预测器实例 - 如果您的图像是白底黑字，设置invert_colors=True
    predictor = TransformerImagePredictor(
        model_path='transformer_image_classifier.pth',
        device='cpu',
        invert_colors=True  # 根据您的图像调整这个参数
    )
    
    # 测试预测
    try:
        result = predictor.predict('/Users/cuijinlong/Documents/workspace_py/fastai_lm/test_images/test_9.png')
        print(predictor.analyze_prediction(result))
    except Exception as e:
        print(f"预测出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n=== 使用方法总结 ===")
    print("""
    基本用法:
    1. 初始化预测器:
       predictor = TransformerImagePredictor('transformer_image_classifier.pth', invert_colors=True)
       # invert_colors=True: 如果图像是白底黑字
       # invert_colors=False: 如果图像是黑底白字（标准MNIST）
    
    2. 预测单张图像:
       result = predictor.predict('path/to/your/image.png')
    
    3. 获取可读结果:
       print(predictor.analyze_prediction(result))
    """)