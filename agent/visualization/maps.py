import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import imageio
import numpy as np
import scipy.ndimage

import cv2

def colorize_draw_agent_and_fit_to_height(
    topdown_map_info: Dict[str, Any], output_height: int
):
    r"""Given the output of the TopDownMap measure, colorizes the map, draws the agent,
    and fits to a desired output height

    :param topdown_map_info: The output of the TopDownMap measure
    :param output_height: The desired output height
    """
    top_down_map = topdown_map_info * 255.0
    # top_down_map = colorize_topdown_map(
    #     top_down_map, topdown_map_info["fog_of_war_mask"]
    # )
    # map_agent_pos = topdown_map_info["agent_map_coord"]
    # top_down_map = draw_agent(
    #     image=top_down_map,
    #     agent_center_coord=map_agent_pos,
    #     agent_rotation=topdown_map_info["agent_angle"],
    #     agent_radius_px=min(top_down_map.shape[0:2]) // 32,
    # )

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map to align with rgb view
    old_h, old_w, _ = top_down_map.shape
    top_down_height = output_height
    top_down_width = int(float(top_down_height) / old_h * old_w)
    # cv2 resize (dsize is width first)
    top_down_map = cv2.resize(
        top_down_map,
        (top_down_width, top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    return top_down_map