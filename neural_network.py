import matplotlib.pyplot as plt
import torch
import torch.nn as nn


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(3648, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 8)
        self.softmax = nn.Softmax(dim=1)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x


def detect_with_nn(incision_dataset, train_dataloader, val_dataloader, train_size, val_size):

    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    n_epochs = 50
    n_images = 5
    # train_on_n_images(n_epochs, n_images, incision_dataset, optimizer, model, criterion)
    # validate_on_n_images(n_images, incision_dataset, model, criterion)

    # train_model(n_epochs, train_size, train_dataloader, optimizer, model, criterion)
    # torch.save(model.state_dict(), "model.pth")
    model = ConvNet()
    model.load_state_dict(torch.load("model.pth"))
    model.eval()
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")
    validate_model(val_size, val_dataloader, model, criterion)


def train_on_n_images(n_epochs, n_images, incision_dataset, optimizer, model, criterion):
    for epoch in range(n_epochs):
        losses = []
        for j in range(n_images):
            img, gray_img, thr_img, quantized_mask, mask, n_stitches = incision_dataset.__getitem__(j)
            optimizer.zero_grad()
            outputs = model(img.unsqueeze(0))
            loss = criterion(outputs, torch.tensor([n_stitches], dtype=torch.long))
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            # print(f"Predicted: {outputs.squeeze()}, Actual: {n_stitches}")
            # print(f"Loss: {loss.item()}")
        print(f"Epoch {epoch + 1}, loss: {sum(losses) / len(losses)}")
    print("Finished training")


def validate_on_n_images(n_images, incision_dataset, model, criterion):
    with torch.no_grad():
        for j in range(n_images):
            img, gray_img, thr_img, quantized_mask, mask, n_stitches = incision_dataset.__getitem__(j)
            outputs = model(img.unsqueeze(0))
            loss = criterion(outputs, torch.tensor([n_stitches], dtype=torch.long))
            # print(f"Predicted idx: {outputs.squeeze().argmax()}, Predicted val: {outputs.squeeze().max()}, Actual: {n_stitches}")
            print(f"Predicted: {outputs.squeeze().argmax()}, Actual: {n_stitches}")


def train_model(n_epochs, train_size, train_dataloader, optimizer, model, criterion):
    for epoch in range(n_epochs):
        total_loss = 0
        for i, data in enumerate(train_dataloader):
            img, gray_img, thr_img, quantized_mask, mask, n_stitches = data
            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, torch.tensor([n_stitches], dtype=torch.long))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, loss: {total_loss / train_size}")
    print("Finished training")


def validate_model(val_size, val_dataloader, model, criterion):
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            img, gray_img, thr_img, quantized_mask, mask, n_stitches = data
            # plt.imshow(img.squeeze().permute(1, 2, 0).numpy())
            # plt.show()
            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == n_stitches).sum().item()
            print(f"Predicted: {outputs.squeeze().argmax().item()}, Actual: {n_stitches.item()}")
            loss = criterion(outputs, torch.tensor([n_stitches], dtype=torch.long))
            total_loss += loss.item()
    accuracy = correct_predictions / val_size
    print(f"Validation loss: {total_loss / val_size}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

