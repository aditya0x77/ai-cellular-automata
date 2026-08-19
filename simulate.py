import torch
import torch.nn as nn
import numpy as np

from vispy import app, scene

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 1, 3, padding=1)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


device = "cuda" if torch.cuda.is_available() else "cpu"

model = NeuralNet().to(device)

model.load_state_dict(
    torch.load("game_of_life_cnn.pth", map_location=device)
)

model.eval()


SIZE = 1000

board = np.random.randint(0, 2, (SIZE, SIZE)).astype(np.float32)

board = torch.from_numpy(board)
board = board.unsqueeze(0).unsqueeze(0).to(device)


# =====================================================
# AI DID THE REST 
# =====================================================

canvas = scene.SceneCanvas(
    keys="interactive",
    size=(800, 800),
    show=True
)

view = canvas.central_widget.add_view()

image = scene.visuals.Image(
    board[0, 0].cpu().numpy(),
    cmap="grays",
    parent=view.scene
)

view.camera = scene.PanZoomCamera(aspect=1)
view.camera.set_range()

generation = 0

@torch.inference_mode()
def update(event):

    global board
    global generation

    board = model(board)

    board = torch.sigmoid(board)

    board = (board > 0.5).float()

    generation += 1

    image.set_data(board[0, 0].cpu().numpy())

    canvas.title = f"Generation {generation}"


timer = app.Timer(interval=0, connect=update, start=True)

app.run()