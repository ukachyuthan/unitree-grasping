"""
Record env 0's rollout state every physics step, for offline rendering.

Isaac Sim's RTX renderer is not needed: a headless run records robot link poses,
the object pose and the grasp outcome, plus the robot's visual meshes (read once
from the USD stage) and the object meshes. scripts/render_recording.py turns the
file into an MP4 with open3d, without Isaac.

File format (.npz):
    meta              JSON string — body names, phase step counts, table, episodes
    body_pos          (T, n_bodies, 3)  env-local, metres
    body_quat         (T, n_bodies, 4)  wxyz
    obj_pose          (T, 7)            active object, env-local pos + wxyz
    step_episode      (T,)              episode index of each frame
    step_exec         (T,)              physics step within the episode
    link_<i>_v/_f     visual mesh of body i, in the link frame
    shape_<j>_v/_f    mesh of meta["shapes"][j], object frame, already scaled
"""

from __future__ import annotations

import json

import numpy as np

from envs._object_registry import shape_asset


def _link_visual_meshes(robot_root: str, body_names: list[str]) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Visual triangle meshes of each robot link, expressed in that link's frame."""
    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.XformCache()
    meshes = {}
    for i, name in enumerate(body_names):
        link = stage.GetPrimAtPath(f"{robot_root}/{name}")
        if not link.IsValid():
            continue
        to_link = cache.GetLocalToWorldTransform(link).GetInverse()
        verts, faces, offset = [], [], 0
        for prim in Usd.PrimRange(link, Usd.TraverseInstanceProxies()):
            if not prim.IsA(UsdGeom.Mesh) or "collision" in str(prim.GetPath()).lower():
                continue
            if UsdGeom.Imageable(prim).ComputePurpose() == UsdGeom.Tokens.guide:
                continue
            mesh = UsdGeom.Mesh(prim)
            pts = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            idx = mesh.GetFaceVertexIndicesAttr().Get()
            if not pts or not counts:
                continue
            # Gf matrices act on row vectors: p_link = p_mesh · M_mesh→world · M_world→link
            m = np.array(cache.GetLocalToWorldTransform(prim) * to_link)
            p = np.c_[np.asarray(pts, dtype=np.float64), np.ones(len(pts))] @ m
            idx = np.asarray(idx)
            start = 0
            for c in counts:   # fan-triangulate quads / n-gons
                for k in range(1, c - 1):
                    faces.append([idx[start] + offset, idx[start + k] + offset, idx[start + k + 1] + offset])
                start += c
            verts.append(p[:, :3])
            offset += len(pts)
        if verts:
            meshes[i] = (np.concatenate(verts).astype(np.float32), np.asarray(faces, dtype=np.int32))
    return meshes


class StateRecorder:
    def __init__(self, env):
        self.u = env.unwrapped
        u = self.u
        self.origin = u.scene.env_origins[0]
        self.body_names = list(u._robot.body_names)
        self.link_meshes = _link_visual_meshes("/World/envs/env_0/Robot", self.body_names)
        print(f"[record] robot visual meshes for {len(self.link_meshes)}/{len(self.body_names)} links")
        self._pos, self._quat, self._obj, self._ep, self._exec = [], [], [], [], []
        self.episodes: list[dict] = []
        self.shapes: list[str] = []

    def begin_episode(self, shape: str):
        u = self.u
        if shape not in self.shapes:
            self.shapes.append(shape)
        ep = {"shape": shape, "spawn_z": float((u._spawn_z[0] - self.origin[2]).item())}
        if u.cfg.two_point_grasp:
            ep["contacts_local"] = [u._contact_left_local[0].tolist(), u._contact_right_local[0].tolist()]
        else:
            ep["contacts_local"] = [u._grasp_target[0].tolist()]
        self.episodes.append(ep)

    def capture(self):
        """Call once per physics step, after scene.update()."""
        u = self.u
        self._pos.append((u._robot.data.body_pos_w[0] - self.origin).cpu().numpy())
        self._quat.append(u._robot.data.body_quat_w[0].cpu().numpy())
        obj = u._active_obj_state()[0, :7].clone()
        obj[:3] -= self.origin
        self._obj.append(obj.cpu().numpy())
        self._ep.append(len(self.episodes) - 1)
        self._exec.append(int(u._exec_step))

    def end_episode(self):
        u = self.u
        self.episodes[-1].update(
            lift_frac=float(u._last_lift_reward[0].item()),
            success=bool(u._last_lift_success[0].item()),
        )

    def save(self, path: str):
        u, cfg = self.u, self.u.cfg
        table = cfg.table
        meta = {
            "body_names": self.body_names,
            "shapes": self.shapes,
            "episodes": self.episodes,
            "physics_dt": float(u.physics_dt),
            "lift_target_m": float(cfg.lift_target_m),
            "table": {"center": [float(v) for v in table.init_state.pos],
                      "size": [float(v) for v in table.spawn.size]},
            # Phase lengths in physics steps, in execution order (see grasp_pose_env_cfg).
            "phases": _phase_table(),
        }
        arrays = {
            "meta": np.array(json.dumps(meta)),
            "body_pos": np.asarray(self._pos, dtype=np.float32),
            "body_quat": np.asarray(self._quat, dtype=np.float32),
            "obj_pose": np.asarray(self._obj, dtype=np.float32),
            "step_episode": np.asarray(self._ep, dtype=np.int32),
            "step_exec": np.asarray(self._exec, dtype=np.int32),
        }
        for i, (v, f) in self.link_meshes.items():
            arrays[f"link_{i}_v"], arrays[f"link_{i}_f"] = v, f
        import trimesh
        for j, shape in enumerate(self.shapes):
            mesh = trimesh.load(str(shape_asset(shape, ".obj")), process=False, force="mesh")
            mesh.apply_scale(cfg.object_scale)
            arrays[f"shape_{j}_v"] = np.asarray(mesh.vertices, dtype=np.float32)
            arrays[f"shape_{j}_f"] = np.asarray(mesh.faces, dtype=np.int32)
        np.savez_compressed(path, **arrays)
        print(f"[record] saved {len(self._pos)} frames, {len(self.episodes)} episodes → {path}")


def _phase_table() -> list[list]:
    from envs import grasp_pose_env_cfg as c
    return [["approach", c.N_APPROACH], ["descend", c.N_DESCEND], ["close", c.N_CLOSE],
            ["lift", c.N_LIFT], ["hold", c.N_HOLD], ["transport", c.N_TRANSPORT],
            ["lower", c.N_LOWER], ["open", c.N_OPEN]]
