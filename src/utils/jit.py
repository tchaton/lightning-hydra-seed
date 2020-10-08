
import os
import torch
import logging

log = logging.getLogger(__name__)


def check_jittable(model, data_module):
    path2model = os.path.join(os.getcwd(), "model.pt")
    torch.jit.save(model.to_torchscript(), path2model)
    model = torch.jit.load(path2model)
    data_module.forward = model.forward
    batch = get_single_batch(data_module)
    _ = data_module.test_step(batch, None)
    
    log.info("Check model is jittable complete.")

def get_single_batch(data_module):
    for batch in data_module.test_dataloader():
        return batch
