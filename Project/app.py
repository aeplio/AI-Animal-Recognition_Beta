from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
import torch
from torchvision import models, transforms
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max-limit

# 确保上传文件夹存在
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# 加载预训练模型
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    model = models.resnet50(pretrained=True)
    model = model.to(device)
    model.eval()
    print('Model loaded successfully')
except Exception as e:
    print(f'Error loading model: {str(e)}')
    raise

# 图像预处理转换
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载类别标签
try:
    with open('imagenet_classes.txt', 'r', encoding='utf-8') as f:
        categories = [line.strip() for line in f.readlines()]
    print(f'Loaded {len(categories)} categories')
except Exception as e:
    print(f'Error loading categories: {str(e)}')
    raise

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print('Error: No file part')
        return jsonify({'error': '没有文件上传'})
    
    file = request.files['file']
    if file.filename == '':
        print('Error: No selected file')
        return jsonify({'error': '没有选择文件'})
    
    if file:
        try:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            print(f'File saved to {filepath}')
            
            # 处理图像
            try:
                image = Image.open(filepath).convert('RGB')
                print('Image opened successfully')
                input_tensor = preprocess(image)
                print('Image preprocessed')
                input_batch = input_tensor.unsqueeze(0).to(device)  # 将输入张量移动到正确的设备上
                print(f'Input batch moved to {device}')
                
                with torch.no_grad():
                    output = model(input_batch)
                print('Model inference completed')
                
                # 获取前5个预测结果
                probabilities = torch.nn.functional.softmax(output[0], dim=0)
                num_classes = len(categories)
                if num_classes == 0:
                    raise ValueError('类别列表为空，请确保正确加载类别文件')
                
                k = min(5, num_classes)  # 确保k不超过类别总数
                top5_prob, top5_catid = torch.topk(probabilities, k)
                
                results = []
                for i in range(k):
                    category_idx = int(top5_catid[i])
                    if 0 <= category_idx < num_classes:
                        category = categories[category_idx]
                        prob = float(top5_prob[i]) * 100
                        print(f'Prediction {i+1}: {category} ({prob:.2f}%)')
                        results.append({
                            'category': category,
                            'probability': prob
                        })
                    else:
                        print(f'Warning: Invalid category index {category_idx}')
                
                if not results:
                    return jsonify({'error': '无法识别图像内容'})
                
                return jsonify({
                    'success': True,
                    'predictions': results
                })
                
            except Exception as e:
                print(f'Error during image processing: {str(e)}')
                return jsonify({'error': f'图像处理错误: {str(e)}'})
                
        except Exception as e:
            print(f'Error during file handling: {str(e)}')
            return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)