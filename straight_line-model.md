import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# Seed for reproducibility
torch.manual_seed(42)

# Generate synthetic data: y = 2x + 3 + noise
true_slope = 2.0
true_intercept = 3.0
num_points = 100

x = torch.unsqueeze(torch.linspace(-10, 10, num_points), dim=1)
y = true_slope * x + true_intercept + 0.5 * torch.randn(x.size())

# Define a simple linear regression model
class LinearModel(nn.Module):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input and one output

    def forward(self, x):
        return self.linear(x)

# Model, loss, optimizer
model = LinearModel()
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 200
losses = []

for epoch in range(num_epochs):
    model.train()

    outputs = model(x)
    loss = criterion(outputs, y)
    losses.append(loss.item())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Plot the loss over epochs
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')

# Plot the fitted line
model.eval()
predicted = model(x).detach().numpy()

plt.subplot(1, 2, 2)
plt.scatter(x.numpy(), y.numpy(), label='Data')
plt.plot(x.numpy(), predicted, color='red', label='Fitted Line')
plt.title('Line Fitting')
plt.legend()
plt.show()

# Print learned parameters
for name, param in model.named_parameters():
    if param.requires_grad:
        print(f"{name}: {param.data.numpy()}")

