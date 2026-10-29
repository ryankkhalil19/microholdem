"""Vanilla CFR+ for a bucketed heads-up river betting game.

Abstraction
    Both players hold a hand strength drawn uniformly from ``[0, 1]`` and see
    only its bucket (``K`` equal-mass buckets). The higher strength wins a
    showdown; equal buckets split. This is a discretized continuous-poker
    model, so the bucket maps directly onto "equity versus the opponent's
    range", which is what the agent measures at runtime.

Betting
    Pot is 1. Each player has ``stack`` pots behind. The out-of-position
    player (OOP) acts first: check or bet (sizes as pot fractions, capped at
    the stack). Facing a bet a player may fold, call or raise once (to
    ``raise_mult`` times the bet, capped at all-in). Facing a raise: fold or
    call. The tree is tiny, so CFR traverses it fully each iteration with all
    bucket pairs vectorized as ``K x K`` matrices.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

F64 = npt.NDArray[np.float64]


@dataclass
class Node:
    key: str  # betting history, e.g. "x b0.5 r"
    player: int  # 0 = OOP, 1 = IP, -1 = terminal
    bets: tuple[float, float]  # chips committed this street by (OOP, IP), in pots
    actions: list[str] = field(default_factory=list)
    children: list[Node] = field(default_factory=list)
    terminal: str = ""  # "fold0", "fold1", "showdown" or "" for decision nodes


@dataclass
class RiverSolution:
    stack: float
    sizes: tuple[float, ...]
    n_buckets: int
    strategies: dict[str, F64]  # node key -> (K, n_actions) average strategy
    actions: dict[str, list[str]]
    exploitability: float  # average best-response gain in pots
    value_oop: float
    iterations: int


def build_tree(stack: float, sizes: tuple[float, ...] = (0.5, 1.0), raise_mult: float = 3.0) -> Node:
    """Build the river betting tree."""

    def bet_options(pot: float, committed: float) -> list[tuple[str, float]]:
        out: list[tuple[str, float]] = []
        seen: set[float] = set()
        for s in sizes:
            amt = min(round(s * pot, 6), round(stack - committed, 6))
            if amt <= 0 or amt in seen:
                continue
            seen.add(amt)
            tag = "a" if amt >= stack - committed - 1e-9 else f"b{s:g}"
            out.append((tag, amt))
        return out

    def facing_bet(key: str, player: int, bets: tuple[float, float], can_raise: bool) -> Node:
        node = Node(key=key, player=player, bets=bets)
        me, opp = bets[player], bets[1 - player]
        # fold
        node.actions.append("f")
        node.children.append(Node(key=key + " f", player=-1, bets=bets, terminal=f"fold{player}"))
        # call
        nb = list(bets)
        nb[player] = opp
        node.actions.append("c")
        node.children.append(Node(key=key + " c", player=-1, bets=(nb[0], nb[1]), terminal="showdown"))
        # raise (once)
        if can_raise and stack - me > opp - me + 1e-9:
            target = min(opp * raise_mult, stack)
            if target > opp + 1e-9:
                nb2 = list(bets)
                nb2[player] = target
                tag = "a" if target >= stack - 1e-9 else "r"
                node.actions.append(tag)
                node.children.append(facing_bet(key + " " + tag, 1 - player, (nb2[0], nb2[1]), can_raise=False))
        return node

    def open_node(key: str, player: int, bets: tuple[float, float]) -> Node:
        node = Node(key=key, player=player, bets=bets)
        pot = 1.0 + bets[0] + bets[1]
        node.actions.append("x")
        if player == 0:
            node.children.append(open_node(key + " x" if key else "x", 1, bets))
        else:
            node.children.append(Node(key=key + " x", player=-1, bets=bets, terminal="showdown"))
        for tag, amt in bet_options(pot, bets[player]):
            nb = list(bets)
            nb[player] = bets[player] + amt
            child_key = (key + " " + tag) if key else tag
            node.children.append(facing_bet(child_key, 1 - player, (nb[0], nb[1]), can_raise=tag != "a"))
            node.actions.append(tag)
        return node

    return open_node("", 0, (0.0, 0.0))


def decision_nodes(root: Node) -> list[Node]:
    out: list[Node] = []

    def walk(n: Node) -> None:
        if n.player >= 0:
            out.append(n)
            for c in n.children:
                walk(c)

    walk(root)
    return out


class RiverCFR:
    def __init__(self, stack: float = 2.0, n_buckets: int = 10, sizes: tuple[float, ...] = (0.5, 1.0)) -> None:
        self.stack = stack
        self.K = n_buckets
        self.sizes = sizes
        self.root = build_tree(stack, sizes)
        self.nodes = decision_nodes(self.root)
        K = n_buckets
        self.regret: dict[str, F64] = {n.key: np.zeros((K, len(n.actions))) for n in self.nodes}
        self.strat_sum: dict[str, F64] = {n.key: np.zeros((K, len(n.actions))) for n in self.nodes}
        self.win = np.zeros((K, K))  # P(player 0 wins | bucket0 = i, bucket1 = j)
        for i in range(K):
            for j in range(K):
                self.win[i, j] = 1.0 if i > j else (0.5 if i == j else 0.0)
        self.prior = np.full(K, 1.0 / K)
        self.iterations = 0

    # ------------------------------------------------------------------
    def strategy(self, key: str) -> F64:
        r = self.regret[key]
        pos = np.maximum(r, 0.0)
        tot = pos.sum(axis=1, keepdims=True)
        return np.where(tot > 0, pos / np.where(tot > 0, tot, 1.0), 1.0 / r.shape[1])

    def average_strategy(self, key: str) -> F64:
        s = self.strat_sum[key]
        tot = s.sum(axis=1, keepdims=True)
        return np.where(tot > 0, s / np.where(tot > 0, tot, 1.0), 1.0 / s.shape[1])

    def terminal_value(self, node: Node) -> F64:
        """Payoff to player 0 as a (K, K) matrix."""
        b0, b1 = node.bets
        if node.terminal == "fold0":
            return np.full((self.K, self.K), -b0)
        if node.terminal == "fold1":
            return np.full((self.K, self.K), 1.0 + b1)
        return self.win * (1.0 + b1) - (1.0 - self.win) * b0

    def _cfr(self, node: Node, reach0: F64, reach1: F64, updating: int, weight: float, strat_fn: str) -> F64:
        if node.player < 0:
            return self.terminal_value(node)
        key = node.key
        sigma = self.strategy(key) if strat_fn == "current" else self.average_strategy(key)
        p = node.player
        values: list[F64] = []
        for a, child in enumerate(node.children):
            if p == 0:
                values.append(self._cfr(child, reach0 * sigma[:, a], reach1, updating, weight, strat_fn))
            else:
                values.append(self._cfr(child, reach0, reach1 * sigma[:, a], updating, weight, strat_fn))
        u = np.zeros((self.K, self.K))
        for a in range(len(values)):
            u += (sigma[:, a][:, None] if p == 0 else sigma[:, a][None, :]) * values[a]
        if strat_fn == "current" and p == updating:
            if p == 0:
                w = (reach1 * self.prior)[None, :]
                for a in range(len(values)):
                    self.regret[key][:, a] += ((values[a] - u) * w).sum(axis=1)
                self.strat_sum[key] += weight * (reach0 * self.prior)[:, None] * sigma
            else:
                w = (reach0 * self.prior)[:, None]
                for a in range(len(values)):
                    self.regret[key][:, a] += (-(values[a] - u) * w).sum(axis=0)
                self.strat_sum[key] += weight * (reach1 * self.prior)[:, None] * sigma
            np.maximum(self.regret[key], 0.0, out=self.regret[key])  # regret matching+
        return u

    def run(self, iterations: int) -> None:
        ones = np.ones(self.K)
        for _ in range(iterations):
            self.iterations += 1
            t = float(self.iterations)
            for player in (0, 1):
                self._cfr(self.root, ones, ones, player, t, "current")

    # ------------------------------------------------------------------
    def game_value(self) -> float:
        ones = np.ones(self.K)
        u = self._cfr(self.root, ones, ones, -1, 0.0, "average")
        return float((self.prior[:, None] * self.prior[None, :] * u).sum())

    def _br(self, node: Node, player: int, reach_opp: F64) -> F64:
        """Value matrix (payoff to player 0) with ``player`` best-responding to the average strategy."""
        if node.player < 0:
            return self.terminal_value(node)
        sigma = self.average_strategy(node.key)
        if node.player != player:
            acc = np.zeros((self.K, self.K))
            for a, child in enumerate(node.children):
                v = self._br(child, player, reach_opp * sigma[:, a])
                acc += (sigma[:, a][:, None] if node.player == 0 else sigma[:, a][None, :]) * v
            return acc
        vals = [self._br(child, player, reach_opp) for child in node.children]
        w = reach_opp * self.prior
        out = np.empty((self.K, self.K))
        if player == 0:
            scores = np.stack([(v * w[None, :]).sum(axis=1) for v in vals])  # (A, K)
            best = scores.argmax(axis=0)
            for i in range(self.K):
                out[i, :] = vals[best[i]][i, :]
        else:
            scores = np.stack([(-v * w[:, None]).sum(axis=0) for v in vals])  # (A, K) payoff to player 1
            best = scores.argmax(axis=0)
            for j in range(self.K):
                out[:, j] = vals[best[j]][:, j]
        return out

    def exploitability(self) -> float:
        ones = np.ones(self.K)
        joint = self.prior[:, None] * self.prior[None, :]
        br0 = float((joint * self._br(self.root, 0, ones)).sum())
        br1 = float((joint * -self._br(self.root, 1, ones)).sum())
        value = self.game_value()
        return 0.5 * ((br0 - value) + (br1 + value))

    def solution(self) -> RiverSolution:
        return RiverSolution(
            stack=self.stack,
            sizes=self.sizes,
            n_buckets=self.K,
            strategies={n.key: self.average_strategy(n.key) for n in self.nodes},
            actions={n.key: list(n.actions) for n in self.nodes},
            exploitability=self.exploitability(),
            value_oop=self.game_value(),
            iterations=self.iterations,
        )


def solve_river(
    stack: float, n_buckets: int = 10, iterations: int = 3000, sizes: tuple[float, ...] = (0.5, 1.0)
) -> RiverSolution:
    solver = RiverCFR(stack=stack, n_buckets=n_buckets, sizes=sizes)
    solver.run(iterations)
    return solver.solution()
