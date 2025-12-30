import cv2


class VideoWriter:
    """
    Simple video writer wrapper.
    Usage:
        writer = VideoWriter("out.mp4", fps=25, frame_size=(w, h))
        writer.write(frame)
        writer.release()
    """

    def __init__(self, output_path, fps, frame_size, codec="mp4v"):
        self.output_path = output_path
        self.fps = fps
        self.frame_size = frame_size

        fourcc = cv2.VideoWriter_fourcc(*codec)
        self.writer = cv2.VideoWriter(
            output_path,
            fourcc,
            fps,
            frame_size
        )

        if not self.writer.isOpened():
            raise RuntimeError("Error: VideoWriter failed to open")

    def write(self, frame):
        """
        Write a single frame.
        Frame must match frame_size (w, h).
        """
        self.writer.write(frame)

    def release(self):
        self.writer.release()
