#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import rospy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from PIL import Image as PILImage
from sensor_msgs.msg import Image

from train_resnet18_regressor import build_transform, create_model, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the trained ResNet18 model online on Jackal camera images and publish Twist commands.",
    )
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best.pt or latest.pt")
    parser.add_argument("--image-topic", type=str, default="/camera/color/image_raw", help="ROS image topic")
    parser.add_argument("--cmd-topic", type=str, default="/ml_cmd_vel_raw", help="Output Twist topic")
    parser.add_argument("--device", type=str, default="auto", help='Device: "auto", "cpu", or "cuda"')
    parser.add_argument("--resize-width", type=int, default=320, help="Resize width used during training")
    parser.add_argument("--resize-height", type=int, default=240, help="Resize height used during training")
    parser.add_argument("--linear-scale", type=float, default=1.0, help="Multiplier applied to predicted linear velocity")
    parser.add_argument("--angular-scale", type=float, default=1.0, help="Multiplier applied to predicted angular velocity")
    parser.add_argument("--max-linear", type=float, default=0.4, help="Clamp linear velocity to [0, max_linear]")
    parser.add_argument("--max-angular", type=float, default=0.8, help="Clamp angular velocity to [-max_angular, max_angular]")
    parser.add_argument("--stale-timeout", type=float, default=0.5, help="Publish zero Twist if no image arrives for this many seconds")
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


class JackalInferenceNode:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.bridge = CvBridge()
        self.device = resolve_device(args.device)
        self.transform = build_transform()
        self.model = self._load_model(args.checkpoint)
        self.last_image_time = time.monotonic()
        self.cmd_pub = rospy.Publisher(args.cmd_topic, Twist, queue_size=1)
        self.image_sub = rospy.Subscriber(args.image_topic, Image, self.image_callback, queue_size=1, buff_size=2**24)
        self.watchdog = rospy.Timer(rospy.Duration(0.1), self.watchdog_callback)

        rospy.loginfo("Jackal inference node started")
        rospy.loginfo(f"checkpoint: {args.checkpoint.resolve()}")
        rospy.loginfo(f"image topic: {args.image_topic}")
        rospy.loginfo(f"cmd topic: {args.cmd_topic}")
        rospy.loginfo(f"device: {self.device}")
        if self.device.type == "cuda":
            device_index = self.device.index if self.device.index is not None else 0
            rospy.loginfo(f"gpu: {torch.cuda.get_device_name(device_index)}")

    def _load_model(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)

        model = create_model(use_pretrained=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        return model

    def image_callback(self, msg: Image) -> None:
        self.last_image_time = time.monotonic()
        try:
            bgr = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(
                rgb,
                (self.args.resize_width, self.args.resize_height),
                interpolation=cv2.INTER_AREA,
            )
            image = PILImage.fromarray(resized)
            tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(tensor).squeeze(0).detach().cpu()

            linear = max(0.0, min(self.args.max_linear, float(output[0].item()) * self.args.linear_scale))
            angular = max(-self.args.max_angular, min(self.args.max_angular, float(output[1].item()) * self.args.angular_scale))
            self.publish_cmd(linear, angular)
        except Exception as exc:
            rospy.logerr_throttle(1.0, f"Inference failed: {exc}")
            self.publish_cmd(0.0, 0.0)

    def watchdog_callback(self, _event) -> None:
        if time.monotonic() - self.last_image_time > self.args.stale_timeout:
            self.publish_cmd(0.0, 0.0)

    def publish_cmd(self, linear: float, angular: float) -> None:
        cmd = Twist()
        cmd.linear.x = linear
        cmd.angular.z = angular
        self.cmd_pub.publish(cmd)


if __name__ == "__main__":
    rospy.init_node("jackal_inference_node")
    node = JackalInferenceNode(parse_args())
    rospy.spin()
