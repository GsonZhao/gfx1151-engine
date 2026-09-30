#!/usr/bin/env python3
"""Print and sanity-check the YaRN inverse-frequency schedule."""

import argparse
import math


def correction_dim(rotary_dim, rotations, theta, original_ctx):
    return rotary_dim * math.log(original_ctx / (rotations * 2.0 * math.pi)) / (
        2.0 * math.log(theta)
    )


def yarn_schedule(factor, original_ctx, beta_fast, beta_slow, theta, rotary_dim):
    low = max(0.0, math.floor(correction_dim(rotary_dim, beta_fast, theta, original_ctx)))
    high = min(rotary_dim - 1.0, math.ceil(correction_dim(rotary_dim, beta_slow, theta, original_ctx)))
    if low == high:
        high += 0.001
    schedule = []
    for i in range(rotary_dim // 2):
        base = theta ** (-2.0 * i / rotary_dim)
        ramp = 0.0 if high <= low else (i - low) / (high - low)
        ramp = max(0.0, min(1.0, ramp))
        schedule.append(base * (1.0 - ramp) + base / factor * ramp)
    attention_factor = 1.0 + 0.1 * math.log(factor) if factor > 1.0 else 1.0
    return low, high, attention_factor, schedule


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--factor", type=float, default=2.0)
    parser.add_argument("--original-ctx", type=int, default=262144)
    parser.add_argument("--beta-fast", type=float, default=32.0)
    parser.add_argument("--beta-slow", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=1e7)
    parser.add_argument("--rotary-dim", type=int, default=64)
    args = parser.parse_args()
    if args.factor < 1.0 or args.original_ctx <= 0 or args.beta_fast <= 0 or args.beta_slow <= 0:
        parser.error("factor >= 1, original-ctx > 0, and beta values > 0 are required")

    low, high, attention_factor, schedule = yarn_schedule(
        args.factor,
        args.original_ctx,
        args.beta_fast,
        args.beta_slow,
        args.theta,
        args.rotary_dim,
    )
    base = [args.theta ** (-2.0 * i / args.rotary_dim) for i in range(args.rotary_dim // 2)]
    if args.factor == 1.0 and any(a != b for a, b in zip(schedule, base)):
        raise SystemExit("FAIL: factor=1 did not preserve the native schedule")
    if args.factor > 1.0 and not all(schedule[i] <= base[i] for i in range(len(base))):
        raise SystemExit("FAIL: YaRN schedule increased an inverse frequency")

    print(f"factor={args.factor:.9g} original_ctx={args.original_ctx}")
    print(f"correction_range={low:.9f}..{high:.9f}")
    print(f"attention_factor={attention_factor:.9f}")
    print("index base_inv_freq yarn_inv_freq ratio")
    for i, (b, value) in enumerate(zip(base, schedule)):
        print(f"{i:5d} {b:.12e} {value:.12e} {value / b:.9f}")
    print("PASS")


if __name__ == "__main__":
    main()
