# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import inspect
import torch
import imageio
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.sys.path.insert(0, "/genloco-loihi/conversion")

import conversion


import argparse
import numpy as np
import os
import random
import time

from motion_imitation.envs import env_builder as env_builder
from motion_imitation.robots import anymal_b_simple, anymal_c_simple,base_robot,mini_cheetah,go1,aliengo,spot,spotmicro,siriusmid_belt,a1
from motion_imitation.real_a1 import a1_robot_real
import datetime

TIMESTEPS_PER_ACTORBATCH = 4096
OPTIM_BATCHSIZE = 256

robot_classes = {
    "laikago" : base_robot.Base_robot,
    "a1" : a1.A1,
    "anymal_b":anymal_b_simple.Anymal_b,
    "anymal_c":anymal_c_simple.Anymal_c,
    "siriusmid_belt":siriusmid_belt.Siriusmid_belt,
    "mini_cheetah":mini_cheetah.Mini_cheetah,
    "go1":go1.Go1,
    "aliengo":aliengo.Aliengo,
    "spot":spot.Spot,
    "spotmicro":spotmicro.SpotMicro,
    "real_a1":a1_robot_real.A1Robot,
    # add new robot class here
}


def test(env, num_episodes=None, robot_name="unknown"):
  episode_count = 0
  o = env.reset()

  frames = []

  conversion.net.to(device)
  conversion.net.eval()

  with torch.no_grad():
    # while episode_count < num_episodes:
    for stepnum in range(500):  # record 1000 frames
      print("Step:", stepnum)
      inputs = torch.tensor(o).float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(0).to(device)  # (B, 451, 1, 1)
      snn_out = conversion.slayer_sdnn(inputs)
     
      a = snn_out.cpu().numpy().squeeze(0).squeeze(-1)


      ann_out = conversion.net(inputs[..., 0, 0, 0])
      ann_out = ann_out.cpu().numpy()

      relative_error_norm = conversion.get_error(ann_out, a)
      print(f"Relative Error in norm 2: {relative_error_norm:.4f}")

      o, r, done, info = env.step(a)
      
      frame = env.render(mode='rgb_array')
      frames.append(frame)
      
      if done:
          o = env.reset()          
          episode_count += 1

  imageio.mimsave('genloco_sim.mp4', frames, fps=20)
  return

def main():
  arg_parser = argparse.ArgumentParser()
  arg_parser.add_argument("--mode", dest="mode", type=str, default="test")
  arg_parser.add_argument("--motion_file", dest="motion_file", type=str, default="motion_imitation/data/motions/a1_pace.txt")
  arg_parser.add_argument("--visualize", dest="visualize", action="store_true", default=False)
  arg_parser.add_argument("--num_test_episodes", dest="num_test_episodes", type=int, default=1)
  arg_parser.add_argument("--robot", dest="robot", type=str, default="a1")
  arg_parser.add_argument("--phase_only", dest="phase_only", action="store_true", default=True)   # TODO what is this
  arg_parser.add_argument("--randomized_robot", dest="randomized_robot", action="store_true", default=False)
  arg_parser.add_argument("--sync_root_rotation", dest="sync_root_rotation", action="store_true", default=False)

  args = arg_parser.parse_args()

  if args.randomized_robot:
    robot_class = None
  else:
    robot_class = robot_classes[args.robot]
  
  os.environ["CUDA_VISIBLE_DEVICES"] = '-1'


  ENABLE_ENV_RANDOMIZER = True
  enable_env_rand = ENABLE_ENV_RANDOMIZER and (args.mode != "test")
  env = env_builder.build_imitation_env(motion_files=[args.motion_file],
                                        num_parallel_envs=1,
                                        mode=args.mode,
                                        enable_randomizer=enable_env_rand,
                                        enable_sync_root_rotation=args.sync_root_rotation,
                                        enable_rendering=args.visualize,
                                        enable_phase_only=args.phase_only,
                                        enable_randomized_robot=args.randomized_robot,
                                        robot_class=robot_class,
                                        visualize=args.visualize)
  


  if args.mode == "test":
      test(env=env,
           num_episodes=args.num_test_episodes,
           robot_name=args.robot)
  else:
      assert False, "Unsupported mode: " + args.mode

  return

if __name__ == '__main__':
  main()
