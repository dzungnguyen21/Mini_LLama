# Mini_LLama

Mini_LLama is a lightweight implementation of the LLama (Large Language Model) architecture, optimized for efficient training and inference on limited hardware. This project is designed for research and experimentation in Natural Language Processing (NLP) and deep learning.

## Features
- Efficient transformer-based architecture
- Customizable model size and training configurations
- Support for fine-tuning on custom datasets
- Lightweight inference for deployment on resource-constrained devices

## Installation
```bash
# Clone the repository
git clone https://github.com/dzungnguyen21/Mini_LLama.git
cd Mini_LLama

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
### Training the Model
```bash
python train.py --config configs/train_config.json
```

### Running Inference
```bash
python infer.py --model checkpoint/model.pth --text "Your input text here"
```

## Configuration
Model and training parameters can be customized in the `configs/` directory. Example:
```json
{
  "model_size": "small",
  "learning_rate": 0.001,
  "batch_size": 32,
  "epochs": 10
}
```

## Dataset Preparation
The dataset should be formatted in JSON or CSV format and placed in the `data/` directory. Modify `data_loader.py` to preprocess your specific dataset.

## Contributing
Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a new branch (`feature-branch`).
3. Commit your changes.
4. Push to your fork and create a pull request.

## Contact
For questions and collaborations, feel free to reach out via GitHub issues or email.

