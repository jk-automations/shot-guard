"""Shot Guard for ComfyUI: checks every generated take against its script and references (dialogue, identity drift,
blocking, on-screen text, hidden cuts), explains what is wrong in plain English, and can re-queue the take with a fix
hint under a credit cap. Clone into ComfyUI/custom_nodes/. CPU only. Apache-2.0."""
if __package__:  # loaded by ComfyUI as a custom node pack (pytest imports this file without a package)
    from .shotguard_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
else:
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = {}, {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
