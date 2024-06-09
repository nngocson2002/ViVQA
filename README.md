<h2 align="center"> <a href="https://github.com/ngocson1042002/ViVQA">Advancing Vietnamese Visual Question Answering
with Transformer and Convolutional Integration</a></h2>
<h5 align="center"> We would be grateful if you like our project and give us a star ⭐ on GitHub.  </h2>

<h5 align="center">

## Getting Started
It currently includes code and models at 
[here](beit3/HCMUS/).
### Installation
```bash
git clone https://github.com/ngocson1042002/ViVQA.git
cd ViVQA/beit3/HCMUS
pip install salesforce-lavis
pip install torchscale timm underthesea efficientnet_pytorch
pip install --upgrade transformers
```
### Run model
We support our work with Hugging Face model [ngocson2002/vivqa-model](https://huggingface.co/ngocson2002/vivqa-model).
```python
from transformers import AutoModel
from transformers import AutoTokenizer
from processor import Processor
from PIL import Image
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModel.from_pretrained("ngocson2002/vivqa-model", trust_remote_code=True).to(device)
processor = Processor()

image = Image.open('./ViVQA/demo/1.jpg').convert('RGB')
question = "màu áo của con chó là gì?"

inputs = processor(image, question, return_tensors='pt')
inputs["image"] = inputs["image"].unsqueeze(0)

model.eval()
with torch.no_grad():
    output = model(**inputs)
    logits = output.logits
    idx = logits.argmax(-1).item()

print("Predicted answer:", model.config.id2label[idx]) # prints: màu đỏ
```
## Authors
- Ngoc-Son Nguyen
([nngocson01042002@gmail.com](mailto:nngocson01042002@gmail.com))
- Van-Son Nguyen
([ngvanson1812@gmail.com](mailto:ngvanson1812@gmail.com))
- Tung Le ([lttung@fit.hcmus.edu.vn](mailto:lttung@fit.hcmus.edu.vn))
### Affiliations
- Faculty of Mathematics and Computer Science, University of Science,
Ho Chi Minh, Vietnam
- Faculty of Information Technology, University of Science, Ho Chi Minh, Vietnam
- Vietnam National University, Ho Chi Minh, Vietnam
## Contact
Contact Ngoc-Son Nguyen ([nngocson01042002@gmail.com](mailto:nngocson01042002@gmail.com)) if you have any questions.

