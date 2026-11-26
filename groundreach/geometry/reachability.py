"""Compute a conservative workspace bound from a fixed or revolute URDF chain."""

import xml.etree.ElementTree as ET

import numpy as np


def workspace_bound(base, urdf, tool_link):
    joints = {
        joint.find("child").get("link"): joint for joint in ET.parse(urdf).getroot().findall("joint")
    }
    chain = []
    link = tool_link
    while link in joints:
        joint = joints[link]
        if joint.get("type") not in {"revolute", "fixed"}:
            raise ValueError("Panda bound requires a chain without prismatic joints")
        origin = joint.find("origin")
        chain.append(np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "))
        link = joint.find("parent").get("link")
    return base + chain[-1], sum(np.linalg.norm(offset) for offset in chain[:-1])
