# Heart Disease Machine Learning 
Brought to you by Roy

This project leverages PyTorch to build and train neural network models for predicting heart disease based on patient data. It provides flexibility in data splitting, model selection, learning rate configuration, and training epochs, allowing users to experiment with different configurations to optimize performance and better understand how changing hyperparameters affects outcome. 

## Table of Contents

- [Features](#features)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
  - [Running the Script](#running-the-script)
  - [Interactive Inputs](#interactive-inputs)
- [Models](#models)
  - [HeartDiseaseModel1](#heartdiseasemodel1)
  - [HeartDiseaseModel2](#heartdiseasemodel2)
  - [HeartDiseaseModel3](#heartdiseasemodel3)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Data Preprocessing**: Splits the dataset into training, validation, and testing sets with customizable ratios.
- **Customizable Models**: Choose between three different neural network architectures.
- **Interactive Configuration**: Configure batch size, learning rate, number of epochs, and training data splits interactively.
- **Performance Evaluation**: Calculates and displays accuracy on training, validation, and test sets.

## Dataset

The project uses the [Heart Disease dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci) which contains various medical attributes of patients and a target variable indicating the presence of heart disease.

### Dataset Structure

| Feature                      | Description                                          |
|------------------------------|------------------------------------------------------|
| age                          | Age of the patient                                   |
| sex                          | Gender (1 = male; 0 = female)                        |
| chest pain type              | Type of chest pain experienced                       |
| resting bp                   | Resting blood pressure (in mm Hg)                    |
| s                            | Serum cholesterol in mg/dl                           |
| fasting blood sugar           | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false) |
| resting ecg                  | Resting electrocardiographic results                 |
| max heart rate               | Maximum heart rate achieved                          |
| exercise angina              | Exercise-induced angina (1 = yes; 0 = no)            |
| oldpeak                      | ST depression induced by exercise relative to rest    |
| ST slope                     | Slope of the peak exercise ST segment                |
| target                       | Presence of heart disease (1 = yes; 0 = no)           |

## Installation

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/heart-disease-pytorch.git
   cd heart-disease-pytorch
   ```

2. **Set Up Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**

   The required packages are listed below. You can install them using `pip`.

   ```bash
   pip install torch torchvision pandas numpy scikit-learn
   ```

   Alternatively, use the provided installation commands in the project.

   ```python
   %pip install torch
   %pip install torchvision
   ```

## Usage

### Running the Script

1. **Ensure the Dataset is Available**

   Place the `heart_disease_dataset.csv` file in the project directory.

2. **Run the Python Script**

   Execute the script using Python.

   ```bash
   python heart_disease_prediction.py
   ```

3. **Interactive Prompts**

   The script will prompt for the following inputs:

   - **Training Split**: Choose between three predefined splits.

     ```
     Training split 1, 2, or 3:
     ```

     - `1`: 80% train, 10% validation, 10% test
     - `2`: 70% train, 15% validation, 15% test
     - `3`: 60% train, 20% validation, 20% test

   - **Batch Size**:

     ```
     Define the batch size:
     ```

     Options based on the comments in the code:
     - `1`: Batch size of 1 (Stochastic Gradient Descent)
     - `50`: Batch size of 50 (Medium size, stable convergence)
     - `250+`: Larger batch sizes (Slower convergence, efficient but may generalize poorly)

   - **Model Selection**:

     ```
     Test 1, 2, or 3:
     ```

     Choose between the three defined models:
     - `1`: HeartDiseaseModel1
     - `2`: HeartDiseaseModel2
     - `3`: HeartDiseaseModel3

   - **Learning Rate**:

     ```
     Learning rate 1, 2, 3, or 4:
     ```

     Options:
     - `1`: 0.01 (Fast learning)
     - `2`: 0.001 (Balanced speed)
     - `3`: 0.0001 (Gradual changes)
     - `4`: 0.00001 (Very fine tuning)

   - **Number of Epochs**:

     ```
     How many epochs:
     ```

     For example, `50`.

4. **Training Progress**

   The script will display epoch and batch loss information during training.

5. **Evaluation Results**

   After training, the script will print the accuracy on training, validation, and test sets.

   ```
   Train accuracy: 85.92%
   Valid accuracy: 85.71%
   Test accuracy: 89.92%
   ```

### Interactive Inputs

The script requires user interaction for configuring the training process. Below are explanations for each prompt:

1. **Training Split Selection**

   Choose how to split your dataset:

   - **Split 1**: 80% training, 10% validation, 10% testing.
   - **Split 2**: 70% training, 15% validation, 15% testing.
   - **Split 3**: 60% training, 20% validation, 20% testing.

2. **Batch Size Definition**

   Define the number of samples processed before the model updates:

   - **1**: Batch size of 1 (Stochastic Gradient Descent).
   - **50**: Batch size of 50 (Medium size, stable convergence).
   - **250+**: Larger batch sizes (Slower convergence, efficient but may generalize poorly).

3. **Model Selection**

   Select one of the three neural network architectures:

   - **Model 1**: A fully connected network with two layers (400 neurons in the first layer).
   - **Model 2**: A smaller network with three layers (64, 32 neurons).
   - **Model 3**: Similar to Model 1 but with an additional layer (400, 50 neurons).

4. **Learning Rate Configuration**

   Choose the learning rate, which controls how much the model is updated:

   - **1**: 0.01 (Fast learning).
   - **2**: 0.001 (Balanced speed).
   - **3**: 0.0001 (Gradual changes).
   - **4**: 0.00001 (Very fine tuning).

5. **Number of Epochs**

   Decide how many times the model will iterate over the entire dataset:

   - **5-10**: May indicate underfitting.
   - **50**: Sufficient for capturing complex patterns without overfitting.
   - **200+**: May lead to overfitting.

## Models

Three different neural network models are defined for experimentation:

### HeartDiseaseModel1

A fully connected network with two layers:

- **fc1**: Input layer to 400 neurons.
- **ReLU activation**.
- **fc2**: 400 neurons to 1 neuron.
- **Sigmoid activation** for binary classification.

```python
class HeartDiseaseModel1(nn.Module):
    def __init__(self):
        super(HeartDiseaseModel1, self).__init__()
        input_size = X_train.shape[1]
        self.fc1 = nn.Linear(input_size, 400)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(400, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return out
```

### HeartDiseaseModel2

A smaller network with three layers:

- **fc1**: 11 input features to 64 neurons.
- **ReLU activation**.
- **fc2**: 64 neurons to 32 neurons.
- **ReLU activation**.
- **fc3**: 32 neurons to 1 neuron.
- **Sigmoid activation**.

```python
class HeartDiseaseModel2(nn.Module):
    def __init__(self):
        super(HeartDiseaseModel2, self).__init__()
        self.fc1 = nn.Linear(11, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x
```

### HeartDiseaseModel3

Similar to Model1 but with an additional layer:

- **fc1**: Input layer to 400 neurons.
- **ReLU activation**.
- **fc2**: 400 neurons to 50 neurons.
- **ReLU activation**.
- **fc3**: 50 neurons to 1 neuron.
- **Sigmoid activation**.

```python
class HeartDiseaseModel3(nn.Module):
    def __init__(self):
        super(HeartDiseaseModel3, self).__init__()
        input_size = X_train.shape[1]
        self.fc1 = nn.Linear(input_size, 400)
        self.fc2 = nn.Linear(400, 50)  # Additional layer
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(50, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        x = self.sigmoid(x)
        return x
```

## Training

The training loop involves the following steps:

1. **Initialize Model**: Based on user selection (Model1, Model2, or Model3).
2. **Move to GPU**: If available, the model is moved to GPU for faster computation.
3. **Configure Learning Rate**: Based on user input.
4. **Set Loss Function and Optimizer**:
   - **Loss Function**: Binary Cross Entropy Loss (`nn.BCELoss`).
   - **Optimizer**: Adam optimizer with the selected learning rate.
5. **Training Loop**:
   - For each epoch:
     - Iterate over each batch in the training data.
     - Perform forward pass, compute loss, perform backward pass, and update weights.
     - Track and print loss for each step.

```python
loss_function = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

losses = []
for epoch in range(NUM_EPOCHS):
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
    for i, (X_batch, y_batch) in enumerate(train_loader):
        if torch.cuda.is_available():
            X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

        X_batch = X_batch.float()

        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = loss_function(outputs, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        print(f'Step {i+1} / {len(train_loader)}, Loss: {loss.item()}')
```

## Evaluation

A function `get_accuracy` calculates the accuracy of the model on the training, validation, and test sets by comparing the predicted labels with the actual labels.

```python
def get_accuracy(data_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            if torch.cuda.is_available():
                X_batch, y_batch = X_batch.cuda(), y_batch.cuda()

            X_batch = X_batch.float()

            outputs = model(X_batch)
            predicted = torch.round(outputs)
            correct += (predicted == y_batch.unsqueeze(1)).sum().item()
            total += y_batch.size(0)

    return (correct / total) * 100

train_accuracy = get_accuracy(train_loader)
valid_accuracy = get_accuracy(valid_loader)
test_accuracy = get_accuracy(test_loader)

print("Train accuracy: ", train_accuracy)
print("Valid accuracy: ", valid_accuracy)
print("Test accuracy: ", test_accuracy)
```

## Results

After training the model for 50 epochs with selected configurations, the accuracy results on the datasets were as follows:

```
Train accuracy:  85.92%
Valid accuracy:  85.71%
Test accuracy:  89.92%
```

These results indicate a strong performance of the model in predicting heart disease presence.

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**
2. **Create a Feature Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [PyTorch](https://pytorch.org/)
- [Heart Disease Dataset](https://www.kaggle.com/datasets/ronitf/heart-disease-uci)
- [Scikit-learn](https://scikit-learn.org/)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)

Special thanks to Kaitlyn who always pushes me to be the best version of myself. 
---