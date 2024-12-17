def render_simulation(sim_id: str, framerate: int, dir: str, notes: str):
    """
    Renders the simulation frames into a video using ffmpeg.

    Parameters:
        sim_id (str): The simulation ID.
        framerate (int): The framerate of the output video.
    """
    import os
    import subprocess

    # Define directories
    frames_dir = os.path.join('outputs', f'{sim_id}', dir)
    output_video = os.path.join('outputs', f'{sim_id}', f'{sim_id}_{notes}.mp4')

    # Ensure the frames directory exists
    if not os.path.isdir(frames_dir):
        raise FileNotFoundError(f"Frames directory not found: {frames_dir}")

    # Build ffmpeg command
    ffmpeg_command = [
        'ffmpeg',
        '-y', # Overwrite output file if it exists
        '-framerate', str(framerate),
        '-i', os.path.join(frames_dir, 'frame_%d.png'),
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]

    # Run ffmpeg
    subprocess.run(ffmpeg_command, check=True)