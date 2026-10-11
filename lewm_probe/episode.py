"""Run one planning episode and record what the planner did and what it was thinking."""

import time

import numpy as np
import torch

from . import common as C


def goal_image(env):
    """The frame the env would show with the agent standing on the target, as (1, 1, H, W, 3)."""
    return env._target_img.cpu().numpy().transpose(1, 2, 0)[None, None].copy()


def replay_plan(model, info, plan, recorder):
    """Re-run the predictor on the single chosen plan with hooks on.

    CEM evaluates 300 candidates x 30 iterations per plan; this isolates the one it picked.
    Returns the imagined latent trajectory, the goal latent, the plan's cost, and for every
    hooked layer the last-token activation at each imagined step.
    """
    single = {k: v.unsqueeze(1) for k, v in info.items() if torch.is_tensor(v)}
    recorder.clear()
    with torch.inference_mode():
        cost = model.get_cost(single, plan.unsqueeze(1))
    acts = {
        name: torch.stack([out[0, -1] if out.ndim == 3 else out[-1] for out in outs]).numpy()
        for name, outs in recorder.records.items()
        if name != 'projector'
    }
    recorder.clear()
    return dict(
        imagined=single['predicted_emb'][0, 0].numpy(),
        goal_emb=single['goal_emb'][0, -1].numpy(),
        cost=float(cost[0, 0]),
        acts=acts,
    )


def expert_goal(world, seed, steps=25):
    """Where the scripted expert gets to after `steps` steps from this seed's start.

    Mirrors the upstream eval, whose goals are frames `goal_offset_steps=25` further along
    an expert trajectory, so they are always reachable within one planning horizon.
    """
    from stable_worldmodel.envs.two_room import ExpertPolicy

    world.reset(seed=seed)
    expert = ExpertPolicy()
    expert.set_env(world.envs)
    for _ in range(steps):
        _, _, term, _, world.infos = world.envs.step(expert.get_action(world.infos))
        if term[0]:
            break
    return world.infos['state'][0, 0].copy()


def dataset_start_goals(n, seed=42, goal_offset=25):
    """(start, goal) agent positions sampled as in the upstream eval: two frames of one
    dataset episode, goal_offset steps apart. Dataset frames equal env renders pixel for pixel."""
    import h5py

    with h5py.File(C.ROOT / '.stable-wm' / 'datasets' / 'tworoom.h5', 'r') as f:
        ep_len, ep_offset = f['ep_len'][:], f['ep_offset'][:]
        rng = np.random.default_rng(seed)
        eps = rng.choice(np.nonzero(ep_len > goal_offset + 1)[0], size=n, replace=False)
        rows = np.sort(ep_offset[eps] + rng.integers(0, ep_len[eps] - goal_offset - 1))
        return [(f['proprio'][r], f['proprio'][r + goal_offset]) for r in rows]


def run_episode(model, world, policy, recorder, seed, max_plans=4, goal_pos=None, start_pos=None):
    """Plan, execute the whole plan open-loop, replan; stop at the goal or after max_plans.

    start_pos and goal_pos override the env's randomly sampled agent and target positions.
    """
    world.reset(seed=seed)
    env = world.envs.envs[0].unwrapped
    if start_pos is not None:
        env._set_state(start_pos)
        world.infos['pixels'] = env.render()[None, None]
        world.infos['state'] = np.asarray(start_pos, dtype=np.float32)[None, None]
    if goal_pos is not None:
        env._set_goal_state(goal_pos)
        world.infos['goal_state'] = np.asarray(goal_pos, dtype=np.float32)[None, None]
    goal = goal_image(env)
    block = C.PLAN['action_block']
    plan_steps = C.PLAN['receding_horizon'] * block

    positions = [world.infos['state'][0, 0].copy()]
    plans, replays, solve_times, plan_start_steps = [], [], [], []
    success = False

    for _ in range(max_plans):
        world.infos['goal'] = goal
        info = policy._prepare_info(world.infos)
        info = {k: v for k, v in info.items() if k in ('pixels', 'goal', 'action')}

        t0 = time.perf_counter()
        out = policy.solver(info)
        solve_times.append(time.perf_counter() - t0)

        plan = out['actions']  # (1, horizon, action_block * 2), normalized action units
        plans.append(plan[0].numpy())
        plan_start_steps.append(len(positions) - 1)
        replays.append(replay_plan(model, info, plan, recorder))

        env_actions = policy.process['action'].inverse_transform(
            plan[0].reshape(plan_steps, -1).numpy()
        )
        for a in env_actions:
            _, _, term, trunc, infos = world.envs.step(a[None].astype(np.float32))
            world.infos = infos
            positions.append(infos['state'][0, 0].copy())
            if term[0]:
                success = True
            if term[0] or trunc[0]:
                break
        if success or trunc[0]:
            break

    return dict(
        seed=seed,
        success=success,
        positions=np.array(positions),
        start=positions[0],
        goal_pos=env.target_position.numpy().copy(),
        plans=np.array(plans),
        plan_start_steps=np.array(plan_start_steps),
        solve_times=np.array(solve_times),
        replays=replays,
        wall=dict(
            axis=env.wall_axis,
            pos=env.wall_pos,
            thickness=env.wall_thickness,
            door_centers=env.door_positions.numpy().copy(),
            door_half_sizes=env.door_sizes.numpy().copy(),
        ),
    )
