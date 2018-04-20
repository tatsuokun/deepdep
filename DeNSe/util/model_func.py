import torch


def save_model(model, save_to: str):
    torch.save(model.state_dict(), save_to)


def load_model(model, model_file: str):
    state_dict = torch.load(model_file)
    model.load_state_dict(state_dict)
