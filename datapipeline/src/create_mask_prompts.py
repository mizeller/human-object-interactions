import os
import cv2
import numpy as np
import gradio as gr
from pprint import pprint


class VideoAnnotator:
    def __init__(self, video_p: str, out_p: str):
        self.video = cv2.VideoCapture(video_p)
        self.frame_count = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_class = 1  # 1=human, 0=object
        self.is_negative = False  # New flag for negative prompts
        self.points = {0: [], 1: []}
        self.point_types = {0: [], 1: []}  # Store whether points are positive (1) or negative (0)
        self.out_p = f"{out_p}/prompts.npy"
        try:
            self.annotations = np.load(self.out_p, allow_pickle=True).item()
            print(f"The following frames are already annoated: {self.annotations.keys()}")
            print(self.annotations)
        except:
            self.annotations = {}

    def load_frame_points(self, frame_idx):
        self.points = {0: [], 1: []}
        self.point_types = {0: [], 1: []}
        if frame_idx in self.annotations:
            for cls in [0, 1]:
                if cls in self.annotations[frame_idx]:
                    self.points[cls] = self.annotations[frame_idx][cls][0].tolist()
                    self.point_types[cls] = self.annotations[frame_idx][cls][1].tolist()

    def get_frame(self, frame_idx):
        self.load_frame_points(frame_idx)
        self.video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.video.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            for cls in [0, 1]:
                for pt, pt_type in zip(self.points[cls], self.point_types[cls]):
                    # Green=positive human, Red=negative human
                    # Blue=positive object, Purple=negative object
                    if cls == 1:  # Human
                        color = (0, 255, 0) if pt_type == 1 else (255, 0, 0)
                    else:  # Object
                        color = (0, 0, 255) if pt_type == 1 else (255, 0, 255)
                    cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, color, -1)
        return frame

    def add_point(self, evt: gr.SelectData, frame_idx):
        self.points[self.current_class].append([evt.index[0], evt.index[1]])
        self.point_types[self.current_class].append(0 if self.is_negative else 1)

        frame_dict = {}
        for cls in [0, 1]:
            if len(self.points[cls]) > 0:
                frame_dict[int(cls)] = (
                    np.array(self.points[cls], dtype=np.float32),
                    np.array(self.point_types[cls], dtype=np.int32),
                )
            else:
                print(f"{cls} not in {self.points.keys()}")

        if frame_dict:
            self.annotations[int(frame_idx)] = frame_dict
            np.save(self.out_p, self.annotations)
            pprint(self.annotations)

        return self.get_frame(frame_idx)

    def set_negative_mode(self, is_negative):
        self.is_negative = is_negative
        return (
            "Positive (active)" if not is_negative else "Positive",
            "Negative (active)" if is_negative else "Negative",
        )

    def clear_points(self, frame_idx):
        self.points = {0: [], 1: []}
        self.point_types = {0: [], 1: []}
        if frame_idx in self.annotations:
            del self.annotations[frame_idx]
            np.save(self.out_p, self.annotations)
            pprint(self.annotations)
        return self.get_frame(frame_idx)

    def set_class(self, cls):
        self.current_class = cls
        return (
            "Human (active)" if cls == 1 else "Human",
            "Object (active)" if cls == 0 else "Object",
        )

    def close(self):
        self.video.release()


def launch(args):
    annotator = VideoAnnotator(video_p=args.video_p, out_p=args.out_p)

    def close_app():
        print("killing process.")
        annotator.close()

        # force exit to continue in shell script...

        os._exit(0)

    with gr.Blocks() as app:
        with gr.Row():
            with gr.Column():
                frame_slider = gr.Slider(
                    0, annotator.frame_count - 1, step=1, label="Frame", value=0
                )
                img = gr.Image(value=annotator.get_frame(0))
                frame_slider.change(annotator.get_frame, frame_slider, img)

                with gr.Row():
                    human_btn = gr.Button("Human (active)", variant="primary")
                    object_btn = gr.Button("Object", variant="primary")

                with gr.Row():
                    positive_btn = gr.Button("Positive (active)", variant="primary")
                    negative_btn = gr.Button("Negative", variant="primary")

                with gr.Row():
                    clear_btn = gr.Button(
                        "Delete Annotations (for current frame)", variant="secondary"
                    )
                    close_btn = gr.Button("Close", variant="stop")

                human_btn.click(
                    lambda: annotator.set_class(1), outputs=[human_btn, object_btn]
                )

                object_btn.click(
                    lambda: annotator.set_class(0), outputs=[human_btn, object_btn]
                )

                positive_btn.click(
                    lambda: annotator.set_negative_mode(False),
                    outputs=[positive_btn, negative_btn]
                )

                negative_btn.click(
                    lambda: annotator.set_negative_mode(True),
                    outputs=[positive_btn, negative_btn]
                )

                img.select(annotator.add_point, [frame_slider], img)
                clear_btn.click(annotator.clear_points, frame_slider, img)
                close_btn.click(fn=close_app)

    app.launch(share=args.online)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--video_p", type=str, default="test/video.mp4")
    parser.add_argument("--out_p", type=str, default="prompts.npy")
    parser.add_argument("--online", action="store_true")
    args = parser.parse_args()

    launch(args)
