from typing import List, Optional, Dict
import numpy as np
import os
import tqdm
import imageio
import logging
import agent.visualization.maps as maps
import cv2

def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logging.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def observations_to_image(observation: Dict, info: Dict) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view_l: List[np.ndarray] = []
    if "rgb" in observation:
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.squeeze(0).cpu().numpy() * 255.0
        else:
            rgb = rgb * 255.0
        egocentric_view_l.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view_l.append(depth_map)

    # add image goal if observation has image_goal info
    if "imagegoal" in observation:
        rgb = observation["imagegoal"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view_l.append(rgb)

    assert (
        len(egocentric_view_l) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view_l, axis=1)

    # draw collision
    if "collisions" in info and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    frame = egocentric_view

    if "occupancy_grid" in info:
        # top_down_map = maps.colorize_draw_agent_and_fit_to_height(
        #     info["occupancy_grid"], egocentric_view.shape[0]
        # )
        top_down_map = info["occupancy_grid"]
        #top_down_map的shape是(128,128,1)，我想要去除最后一个维度
        top_down_map = top_down_map.squeeze()
        top_down_map = cv2.resize(
        top_down_map,
        (egocentric_view.shape[0], egocentric_view.shape[0]),
        interpolation=cv2.INTER_CUBIC,
        )
        top_down_map = cv2.cvtColor(top_down_map, cv2.COLOR_GRAY2RGB)
        top_down_map = top_down_map * 255.0
        frame = np.concatenate((egocentric_view, top_down_map), axis=1)
    return frame

def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view