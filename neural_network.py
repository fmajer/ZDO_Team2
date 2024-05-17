from typing import Callable
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from torchvision import transforms
import copy

from tqdm import tqdm


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

    def predict(self, img):
        resize_img = transforms.Resize(size=(50, 180))
        img_tensor = resize_img(torch.from_numpy(img).permute(2, 0, 1)).type(torch.float32)
        outputs = self.forward(img_tensor.unsqueeze(0))
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item()


def load_nn(path):
    model = ConvNet()
    model.load_state_dict(torch.load(path))
    model.eval()
    return model


def train_nn(
    n_epochs, learning_rate, incision_dataset, train_dataloader, val_dataloader, train_size, val_size, device, path_func: Callable[[float], str]
):
    model = ConvNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # n_images = 5
    # train_on_n_images(n_epochs, n_images, incision_dataset, optimizer, model, criterion)
    # validate_on_n_images(n_images, incision_dataset, model)

    best_m, accuracy = train_model(n_epochs, train_size, train_dataloader, val_size, val_dataloader, optimizer, device, model, criterion)

    torch.save(best_m.state_dict(), path_func(accuracy))
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Number of parameters: {total_params}")


def train_on_n_images(n_epochs, n_images, incision_dataset, optimizer, model, criterion):
    for epoch in range(n_epochs):
        losses = []
        for j in range(n_images):
            img, mask, n_stitches = incision_dataset.__getitem__(j)
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


def validate_on_n_images(n_images, incision_dataset, model):
    with torch.no_grad():
        for j in range(n_images):
            img, mask, n_stitches = incision_dataset.__getitem__(j)
            outputs = model(img.unsqueeze(0))
            # print(f"Predicted idx: {outputs.squeeze().argmax()}, "
            #       f"Predicted val: {outputs.squeeze().max()}, Actual: {n_stitches}")
            print(f"Predicted: {outputs.squeeze().argmax()}, Actual: {n_stitches}")


def train_model(n_epochs, train_size, train_dataloader, val_size, val_dataloader, optimizer, device, model, criterion):
    best_val_loss = float("inf")
    best_model = None
    progress = tqdm(range(n_epochs))

    fig1 = plt.figure()
    ax1 = fig1.add_subplot()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot()

    fig3 = plt.figure()
    ax3 = fig3.add_subplot()
    ax = [ax1, ax2, ax3]

    x_data = []
    y1_data = []
    y2_data = []
    y3_data = []
    y_data = [y1_data, y2_data, y3_data]
    colors = ["red", "green", "blue"]

    for epoch in progress:
        total_loss = 0
        for i, data in enumerate(train_dataloader):
            img, mask, n_stitches = data

            img = img.to(device)
            n_stitches = n_stitches.to(device)

            optimizer.zero_grad()
            outputs = model(img)
            loss = criterion(outputs, n_stitches)
            # loss = criterion(outputs, torch.tensor([n_stitches], dtype=torch.long)) # for batch_size=1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, accuracy = validate_model(val_size, val_dataloader, device, model, criterion)
        progress.set_description(f"Train loss: {(total_loss / train_size):.5f}, Validation loss: {val_loss:.5f}, Accuracy: {accuracy:.2%}")

        y_data[0].append(total_loss / train_size)
        y_data[1].append(val_loss)
        y_data[2].append(accuracy)
        x_data.append(epoch)

        if epoch % 25 == 0:
            for i, data in enumerate(y_data):
                ax[i].scatter(x_data, data, color=colors[i])
                data.clear()

            x_data.clear()

        plt.pause(0.05)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = copy.deepcopy(model)

    print("Finished training")
    return best_model, accuracy


def validate_model(val_size, val_dataloader, device, model, criterion) -> tuple[float, float]:
    total_loss = 0
    correct_predictions = 0
    with torch.no_grad():
        for i, data in enumerate(val_dataloader):
            img, mask, n_stitches = data
            img = img.to(device)
            n_stitches = n_stitches.to(device)

            outputs = model(img)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == n_stitches).sum().item()
            loss = criterion(outputs, n_stitches)
            # loss = criterion(outputs, torch.tensor([n_stitches], dtype=torch.long)) # for batch_size=1
            total_loss += loss.item()
            # print(f"Predicted: {outputs.squeeze().argmax().item()}, Actual: {n_stitches.item()}")
            # plt.imshow(img.squeeze().permute(1, 2, 0).numpy())
            # plt.show()
    accuracy = correct_predictions / val_size
    # print(f"Validation loss: {total_loss / val_size}")
    # print(f"Accuracy: {accuracy * 100:.2f}%")
    return total_loss / val_size, accuracy
