
import torch
import numpy as np
import cv2

import torchvision.transforms as trans

image_to_tensor_transform = trans.Compose([
  trans.ToTensor(),
  trans.Resize((64,64)),
  trans.Normalize((0.5, 0.5, 0.5), (1, 1, 1))
])




def image_to_tensor(observation):
  if isinstance(observation, (list, tuple)):

    for i in range(len(observation)):
      observation[i] = image_to_tensor_transform(np.ascontiguousarray(observation[i]))
    observation = torch.stack(observation)
  else:
    observation = image_to_tensor_transform(np.ascontiguousarray(observation))

  return observation


def _random_action(spec):
  return torch.from_numpy(np.random.uniform(spec.minimum, spec.maximum, spec.shape).astype(np.float32))


def stack_batch(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])


def bottle(f, x_tuple):
  x_sizes = tuple(map(lambda x: x.size(), x_tuple))
  y = f(*map(lambda x: x[0].view(x[1][0] * x[1][1], *x[1][2:]), zip(x_tuple, x_sizes)))
  y_size = y.size()
  return y.view(x_sizes[0][0], x_sizes[0][1], *y_size[1:])