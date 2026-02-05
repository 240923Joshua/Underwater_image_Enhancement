from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("logs")
writer.add_scalar("test/value", 1, 0)
writer.flush()
writer.close()
print("done")
