from models.pointnet_encoder import PointNetEncoder

__all__ = ["PointNetEncoder", "PointNetActorCritic"]


def __getattr__(name: str):
    if name == "PointNetActorCritic":
        from models.grasp_actor_critic import PointNetActorCritic

        return PointNetActorCritic
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
