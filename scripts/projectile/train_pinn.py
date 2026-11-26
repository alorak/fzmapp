"""Train a physics-informed neural network (PINN) for projectile motion with quadratic air drag.

Physics (y up, launch from the origin):
    a = -g * y_hat - k * |v| * v        k = rho * Cd * A / (2 m)   [1/m]

With length L = v0^2 / g and time T = v0 / g the equations become dimensionless:
    X'' = -kappa * |V| * X'
    Y'' = -1 - kappa * |V| * Y'
    X(0) = Y(0) = 0,  X'(0) = cos(theta),  Y'(0) = sin(theta),  kappa = k * v0^2 / g
so one network covers every launch speed: inputs (tau, theta, kappa), outputs (X, Y).

The network output is wrapped so the initial conditions hold exactly and the drag-free case is the
textbook parabola:
    X = cos(theta) * (tau + kappa * tau^2 * Nx(tau, theta, kappa))
    Y = tau * sin(theta) - tau^2 / 2  + kappa * tau^2 * Ny(tau, theta, kappa)
Training uses only the residual of the equations of motion at random points (no trajectory data).
The result is checked against a classical Runge-Kutta solution and exported to ONNX for the browser.

Run from the repository root:  python -m scripts.projectile.train_pinn
"""

import json
import math
import os
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import onnx
import onnxruntime as ort
import optax
from onnx import TensorProto, helper, numpy_helper

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "egik-atis.onnx"
META_PATH = ROOT / "models" / "egik-atis.json"

THETA_MIN, THETA_MAX = math.radians(5), math.radians(85)
KAPPA_MAX = 3.5          # k = 0.02 1/m at v0 = 40 m/s gives kappa = 3.26
TAU_SPAN = 2.1           # the drag-free flight lasts 2 sin(theta) in units of T
LAYERS = [3, 48, 48, 48, 2]
STEPS = int(os.environ.get("PINN_STEPS", 40000))
BATCH = 4096
SEED = 7


# ---------- network ----------
def init_params(key):
    params = []
    for fan_in, fan_out in zip(LAYERS[:-1], LAYERS[1:]):
        key, sub = jax.random.split(key)
        limit = math.sqrt(6 / (fan_in + fan_out))
        params.append((jax.random.uniform(sub, (fan_in, fan_out), minval=-limit, maxval=limit), jnp.zeros(fan_out)))
    return params


def features(tau, theta, kappa):
    """Inputs scaled to roughly [-1, 1]."""
    u = tau / (TAU_SPAN * jnp.sin(theta))
    th = (theta - (THETA_MIN + THETA_MAX) / 2) / ((THETA_MAX - THETA_MIN) / 2)
    ka = kappa / KAPPA_MAX * 2 - 1
    return jnp.stack([2 * u - 1, th, ka])


def mlp(params, x):
    for w, b in params[:-1]:
        x = jnp.tanh(x @ w + b)
    w, b = params[-1]
    return x @ w + b


def position(params, tau, theta, kappa):
    n = mlp(params, features(tau, theta, kappa))
    # The x equation is linear in x, so cos(theta) factors out exactly; the network then learns a
    # relative correction, which keeps steep throws (tiny range) as accurate as flat ones.
    x = jnp.cos(theta) * (tau + kappa * tau**2 * n[0])
    y = tau * jnp.sin(theta) - tau**2 / 2 + kappa * tau**2 * n[1]
    return jnp.stack([x, y])


# ---------- physics loss ----------
def residual(params, tau, theta, kappa):
    f = lambda t: position(params, t, theta, kappa)
    velocity = lambda t: jax.jvp(f, (t,), (1.0,))[1]
    v, a = jax.jvp(velocity, (tau,), (1.0,))
    speed = jnp.sqrt(v[0] ** 2 + v[1] ** 2 + 1e-12)
    # The ansatz is exact without drag, so the raw residual shrinks with kappa. Dividing by
    # (kappa + 0.1) keeps light drag from being ignored; dividing x by cos(theta) undoes its scale.
    r = jnp.stack([(a[0] + kappa * speed * v[0]) / jnp.cos(theta), a[1] + 1 + kappa * speed * v[1]])
    return r / (kappa + 0.1)


batched_residual = jax.vmap(residual, in_axes=(None, 0, 0, 0))


def sample(key, n):
    k1, k2, k3 = jax.random.split(key, 3)
    theta = jax.random.uniform(k1, (n,), minval=THETA_MIN, maxval=THETA_MAX)
    kappa = KAPPA_MAX * jax.random.uniform(k2, (n,))
    tau = TAU_SPAN * jnp.sin(theta) * jax.random.uniform(k3, (n,))
    return tau, theta, kappa


def loss(params, batch):
    r = batched_residual(params, *batch)
    return jnp.mean(jnp.sum(r**2, axis=1))


# ---------- reference solution ----------
def rk4(theta, kappa, dt=1e-3, tau_end=None):
    """Classical Runge-Kutta solution of the dimensionless equations until the projectile lands."""
    def rhs(s):
        x, y, vx, vy = s
        speed = math.hypot(vx, vy)
        return np.array([vx, vy, -kappa * speed * vx, -1 - kappa * speed * vy])

    s = np.array([0.0, 0.0, math.cos(theta), math.sin(theta)])
    t, path = 0.0, [(0.0, 0.0, 0.0)]
    while True:
        k1 = rhs(s); k2 = rhs(s + dt / 2 * k1); k3 = rhs(s + dt / 2 * k2); k4 = rhs(s + dt * k3)
        new = s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        if new[1] < 0:
            frac = s[1] / (s[1] - new[1])
            land = s + frac * (new - s)
            path.append((t - dt + frac * dt, land[0], 0.0))
            return np.array(path)
        s = new
        path.append((t, s[0], s[1]))


def summary(path):
    t, x, y = path[:, 0], path[:, 1], path[:, 2]
    return {"range": float(x[-1]), "height": float(y.max()), "time": float(t[-1])}


def model_summary(predict, theta, kappa, steps=2000):
    tau = np.linspace(0, TAU_SPAN * math.sin(theta), steps)
    xy = predict(tau, np.full_like(tau, theta), np.full_like(tau, kappa))
    y = xy[:, 1]
    land = np.nonzero((y[1:] < 0) & (y[:-1] >= 0))[0]
    i = land[0] if len(land) else len(y) - 2
    frac = y[i] / (y[i] - y[i + 1])
    return {
        "range": float(xy[i, 0] + frac * (xy[i + 1, 0] - xy[i, 0])),
        "height": float(y.max()),
        "time": float(tau[i] + frac * (tau[i + 1] - tau[i])),
    }


# ---------- material for the explanation page ----------
# A football at 25 m/s and 45 degrees: k = 0.013 1/m gives kappa = 0.013 * 25^2 / 9.81.
DEMO = {"v0": 25.0, "angle_degrees": 45, "k": 0.013}
DEMO_THETA = math.radians(DEMO["angle_degrees"])
DEMO_KAPPA = DEMO["k"] * DEMO["v0"] ** 2 / 9.81
SNAPSHOT_STEPS = [0, 200, 1000, 4000, 15000]


def demo_path(params, points=48):
    tau = jnp.linspace(0, TAU_SPAN * math.sin(DEMO_THETA), points)
    xy = jax.vmap(lambda t: position(params, t, DEMO_THETA, DEMO_KAPPA))(tau)
    return [[round(float(x), 4), round(float(y), 4)] for x, y in np.asarray(xy)]


def residual_check(params, tau=0.6):
    """What the network's acceleration is at one moment of the demo throw, and what physics demands."""
    f = lambda t: position(params, t, DEMO_THETA, DEMO_KAPPA)
    velocity = lambda t: jax.jvp(f, (t,), (1.0,))[1]
    v, a = jax.jvp(velocity, (jnp.float32(tau),), (1.0,))
    speed = float(jnp.hypot(v[0], v[1]))
    wanted = [-DEMO_KAPPA * speed * float(v[0]), -1 - DEMO_KAPPA * speed * float(v[1])]
    return {
        "tau": tau,
        "network": [round(float(a[0]), 4), round(float(a[1]), 4)],
        "physics": [round(wanted[0], 4), round(wanted[1], 4)],
    }


# ---------- ONNX export ----------
def export_onnx(params, path):
    """Writes the whole model, feature scaling and output formula included, as a small ONNX graph.

    Input  "inputs":   float32 [N, 3] = (tau, theta [rad], kappa)
    Output "position": float32 [N, 2] = (X, Y) in units of v0^2 / g
    """
    nodes, inits = [], []

    def const(name, value):
        inits.append(numpy_helper.from_array(np.asarray(value, dtype=np.float32), name))
        return name

    def node(op, inputs, output, **attrs):
        nodes.append(helper.make_node(op, inputs, [output], **attrs))
        return output

    def index(name, value):
        inits.append(numpy_helper.from_array(np.asarray([value], dtype=np.int64), name))
        return name

    i0, i1, i2 = index("i0", 0), index("i1", 1), index("i2", 2)
    tau = node("Gather", ["inputs", i0], "tau", axis=1)        # [N, 1] after Unsqueeze below
    theta = node("Gather", ["inputs", i1], "theta", axis=1)
    kappa = node("Gather", ["inputs", i2], "kappa", axis=1)
    sin_t = node("Sin", [theta], "sin_t")
    cos_t = node("Cos", [theta], "cos_t")
    # Features, matching features() above.
    u = node("Div", [tau, node("Mul", [sin_t, const("span", [TAU_SPAN])], "span_sin")], "u")
    f0 = node("Sub", [node("Mul", [u, const("two", [2.0])], "u2"), const("one", [1.0])], "f0")
    th_c = node("Sub", [theta, const("th_mid", [(THETA_MIN + THETA_MAX) / 2])], "th_c")
    f1 = node("Div", [th_c, const("th_half", [(THETA_MAX - THETA_MIN) / 2])], "f1")
    f2 = node("Sub", [node("Mul", [kappa, const("ka_scale", [2 / KAPPA_MAX])], "ka_s"), const("one_b", [1.0])], "f2")
    x = node("Concat", [f0, f1, f2], "features", axis=1)
    for index, (w, b) in enumerate(params):
        x = node("Add", [node("MatMul", [x, const(f"W{index}", np.array(w))], f"mm{index}"), const(f"b{index}", np.array(b))], f"z{index}")
        if index < len(params) - 1:
            x = node("Tanh", [x], f"h{index}")
    nx = node("Gather", [x, i0], "nx", axis=1)
    ny = node("Gather", [x, i1], "ny", axis=1)
    tau2 = node("Mul", [tau, tau], "tau2")
    k_tau2 = node("Mul", [kappa, tau2], "k_tau2")
    out_x = node("Mul", [node("Add", [tau, node("Mul", [k_tau2, nx], "corr_x")], "xi"), cos_t], "out_x")
    free_y = node("Sub", [node("Mul", [tau, sin_t], "tau_sin"), node("Mul", [tau2, const("half", [0.5])], "half_tau2")], "free_y")
    out_y = node("Add", [free_y, node("Mul", [k_tau2, ny], "corr_y")], "out_y")
    node("Concat", [out_x, out_y], "position", axis=1)

    # Gather with a 1-element index keeps the axis, so every intermediate is [N, 1].
    graph = helper.make_graph(
        nodes, "egik_atis_pinn",
        [helper.make_tensor_value_info("inputs", TensorProto.FLOAT, ["N", 3])],
        [helper.make_tensor_value_info("position", TensorProto.FLOAT, ["N", 2])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="fzmapp")
    model.ir_version = 8
    onnx.checker.check_model(model)
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, path)


def main():
    key = jax.random.PRNGKey(SEED)
    key, init_key = jax.random.split(key)
    params = init_params(init_key)
    schedule = optax.cosine_decay_schedule(2e-3, STEPS, alpha=0.01)
    optimizer = optax.adam(schedule)
    state = optimizer.init(params)

    @jax.jit
    def step(params, state, key):
        batch = sample(key, BATCH)
        value, grads = jax.value_and_grad(loss)(params, batch)
        updates, state = optimizer.update(grads, state, params)
        return optax.apply_updates(params, updates), state, value

    start = time.time()
    history, window = [], []
    snapshots = []
    first_check = residual_check(params)
    for index in range(STEPS):
        if index in SNAPSHOT_STEPS:
            snapshots.append({"step": index, "path": demo_path(params)})
        key, sub = jax.random.split(key)
        params, state, value = step(params, state, sub)
        window.append(float(value))
        if (index + 1) % 250 == 0:
            # Mean loss over the last 250 steps, for the learning curve on the explanation page.
            history.append([index + 1, float(f"{np.mean(window):.4g}")])
            window = []
        if index % 2000 == 0 or index == STEPS - 1:
            print(f"step {index:6d}  physics loss {float(value):.3e}  {time.time() - start:6.0f}s", flush=True)
    snapshots.append({"step": STEPS, "path": demo_path(params)})

    export_onnx(params, MODEL_PATH)

    # Check the exported model against JAX and against the Runge-Kutta solution.
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    def onnx_predict(tau, theta, kappa):
        inputs = np.stack([tau, theta, kappa], axis=1).astype(np.float32)
        return session.run(None, {"inputs": inputs})[0]
    jax_predict = jax.jit(jax.vmap(lambda t, th, ka: position(params, t, th, ka)))
    probe = sample(jax.random.PRNGKey(1), 4096)
    gap = float(np.max(np.abs(onnx_predict(*[np.asarray(a) for a in probe]) - np.asarray(jax_predict(*probe)))))

    angles = list(range(10, 86, 5))
    kappas = [float(k) for k in np.linspace(0, KAPPA_MAX, 8)]
    errors = {"range": [], "height": [], "time": []}
    for theta_deg in angles:
        for kappa in kappas:
            theta = math.radians(theta_deg)
            ref = summary(rk4(theta, kappa))
            got = model_summary(onnx_predict, theta, kappa)
            for name in errors:
                errors[name].append(abs(got[name] - ref[name]) / ref[name])
    metrics = {name: {"mean_percent": 100 * float(np.mean(v)), "max_percent": 100 * float(np.max(v))} for name, v in errors.items()}
    metrics["cases"] = len(errors["range"])
    metrics["range_grid"] = {
        "angles_degrees": angles,
        "kappas": [round(k, 3) for k in kappas],
        "percent": [[round(100 * e, 3) for e in errors["range"][i * len(kappas):(i + 1) * len(kappas)]] for i in range(len(angles))],
    }

    reference = rk4(DEMO_THETA, DEMO_KAPPA)
    pick = np.linspace(0, len(reference) - 1, 48).round().astype(int)
    tau_example = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    example_out = onnx_predict(np.array(tau_example), np.full(6, DEMO_THETA), np.full(6, DEMO_KAPPA))
    explanation = {
        "demo": {**DEMO, "kappa": round(DEMO_KAPPA, 4)},
        "loss_history": history,
        "snapshots": snapshots,
        "runge_kutta_path": [[round(float(x), 4), round(float(y), 4)] for _, x, y in reference[pick]],
        "residual_check": {"start": first_check, "end": residual_check(params)},
        "example": [
            {"tau": t, "x": round(float(x), 4), "y": round(float(y), 4)} for t, (x, y) in zip(tau_example, example_out)
        ],
    }
    test_loss = float(loss(params, sample(jax.random.PRNGKey(2), 20000)))

    meta = {
        "model": "egik-atis.onnx",
        "description": "PINN for projectile motion with quadratic air drag; inputs (tau, theta, kappa), outputs (X, Y) in units of v0^2/g, time in units of v0/g.",
        "inputs": {"tau": "t * g / v0", "theta": "launch angle in radians", "kappa": "k * v0^2 / g"},
        "ranges": {"theta_degrees": [5, 85], "kappa": [0, KAPPA_MAX], "tau": f"0 .. {TAU_SPAN} * sin(theta)"},
        "layers": LAYERS,
        "parameters": int(sum(w.size + b.size for w, b in params)),
        "training": {"steps": STEPS, "batch": BATCH, "loss": "residual of the equations of motion only", "final_physics_loss": test_loss, "seconds": round(time.time() - start)},
        "check_against_runge_kutta": metrics,
        "onnx_vs_jax_max_abs_difference": gap,
        "explanation": explanation,
    }
    META_PATH.write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "explanation"}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
