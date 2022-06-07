import torch
import torchvision.transforms as T

from datasets import SomethingSomethingR3M

task_names = ['push_left_right']
lr = 0.001
debug = False

def main():
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=False).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss()

    train_data = SomethingSomethingR3M(task_names, train=True, debug=debug)
    valid_data = SomethingSomethingR3M(task_names, train=False, debug=debug)
    transform = T.Compose(T.Resize(224))


    print('Creating data loaders...')
    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=32, shuffle=True,
        num_workers=0 if debug else 8, drop_last=True)
    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=32, shuffle=True,
        num_workers=0 if debug else 8, drop_last=True)

    print('Creating data loaders: done')
    model.train()
    for epoch in range(10):
        for step, data in enumerate(train_queue):
            r3m_embedding, task, delta_joints_smpl = data
            r3m_embedding, task, delta_joints_smpl = r3m_embedding.to(device), task.to(device), delta_joints_smpl.to(
                device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(transform(r3m_embedding))
            loss = loss_func(outputs, delta_joints_smpl)




if __name__ == '__main__':
    main()
