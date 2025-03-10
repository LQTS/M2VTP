import os
asset_abs_path = os.path.dirname(os.path.abspath(__file__))
asset_rel_path = os.path.dirname(os.path.relpath(__file__))
XML_DIR = os.path.join(asset_abs_path, 'xml')
ENV_PY_DIR = os.path.join(asset_rel_path, 'env_py').replace('/', '.')

# add envs
from .env_py.base_env import BaseEnv
from .env_py.shadow_hand import ShadowHandEnv

# register envs
from gym import register
register(id='ShadowHand' + '-v0',
         entry_point='%s.%s:%s' % (ENV_PY_DIR, 'shadow_hand', 'ShadowHandEnv'),
         max_episode_steps=200,
         kwargs={'xml_path': XML_DIR + '/shadow_hand/right_hand_scene_camera.xml'})
