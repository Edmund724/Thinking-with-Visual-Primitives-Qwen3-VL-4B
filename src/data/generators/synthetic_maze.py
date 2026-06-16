"""Maze dataset generator with DFS exploration logs, 50% unsolvable mazes, and 3-step thinking protocol."""

import random
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw

from ...utils.constants import MAZE_SOLVABLE_RATIO
from ...utils.thinking_verifier import filter_verified_samples
from ..formatters.primitive_formatter import format_point, normalize_coordinate


def generate_maze_grid(rows: int, cols: int) -> np.ndarray:
    """Generate a perfect maze using recursive backtracking algorithm.

    Returns:
        Grid where 1 = path, 0 = wall.
    """
    grid = np.zeros((rows * 2 + 1, cols * 2 + 1), dtype=np.uint8)

    def carve(r: int, c: int):
        grid[r * 2 + 1, c * 2 + 1] = 1
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        random.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr * 2 + 1, nc * 2 + 1] == 0:
                grid[r * 2 + 1 + dr, c * 2 + 1 + dc] = 1
                carve(nr, nc)

    carve(0, 0)
    return grid


def find_start_end(maze_grid: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """Find start (top-left path cell) and end (bottom-right path cell)."""
    h, w = maze_grid.shape
    start = None
    for y in range(h):
        for x in range(w):
            if maze_grid[y, x] == 1:
                start = (x, y)
                break
        if start:
            break

    end = None
    for y in range(h - 1, -1, -1):
        for x in range(w - 1, -1, -1):
            if maze_grid[y, x] == 1:
                end = (x, y)
                break
        if end:
            break

    return start, end


def solve_maze_dfs(
    maze_grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> List[Tuple[int, int]]:
    """Solve maze using DFS, return path from start to end."""
    h, w = maze_grid.shape
    visited = set()
    stack = [(start, [start])]

    while stack:
        (x, y), path = stack.pop()
        if (x, y) == end:
            return path
        if (x, y) in visited:
            continue
        visited.add((x, y))

        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze_grid[ny, nx] == 1 and (nx, ny) not in visited:
                stack.append(((nx, ny), path + [(nx, ny)]))

    return []


def break_path_middle(
    maze_grid: np.ndarray,
    solution_path: List[Tuple[int, int]],
    num_walls: int = 3,
) -> np.ndarray:
    """Make maze unsolvable by placing walls around the middle of the solution path."""
    new_grid = maze_grid.copy()
    if len(solution_path) < 4:
        return new_grid

    mid_idx = len(solution_path) // 2
    for i in range(max(0, mid_idx - num_walls // 2), min(len(solution_path), mid_idx + num_walls // 2 + 1)):
        x, y = solution_path[i]
        new_grid[y, x] = 0

    return new_grid


def add_backtracking_trap(
    maze_grid: np.ndarray,
    solution_path: List[Tuple[int, int]],
    trap_length: int = 3,
) -> np.ndarray:
    """Add a dead-end corridor adjacent to the solution path.

    The solver must enter the trap, reach the dead end, and backtrack before
    continuing along the real solution. This produces explicit backtracking
    behavior in the exploration log.

    Returns:
        Modified maze grid with the trap carved in.
    """
    new_grid = maze_grid.copy()
    if len(solution_path) < 4:
        return new_grid

    h, w = new_grid.shape

    # Pick a cell on the solution path that is not the start/end and has
    # at least one wall neighbor that can be turned into a dead-end corridor.
    candidates = []
    for i in range(1, len(solution_path) - 1):
        x, y = solution_path[i]
        # Neighbors that are currently walls
        wall_neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and new_grid[ny, nx] == 0:
                wall_neighbors.append((nx, ny, dx, dy))
        if wall_neighbors:
            candidates.append((x, y, wall_neighbors))

    if not candidates:
        return new_grid

    import random
    x, y, wall_neighbors = random.choice(candidates)
    nx, ny, dx, dy = random.choice(wall_neighbors)

    # Carve the first cell of the trap.
    new_grid[ny, nx] = 1

    # Continue straight as far as possible to create a corridor.
    carved = 1
    cx, cy = nx, ny
    while carved < trap_length:
        fx, fy = cx + dx, cy + dy
        if 0 <= fx < w and 0 <= fy < h and new_grid[fy, fx] == 0:
            # Only carve if it doesn't connect to another path cell.
            side_neighbors = [
                (fx + sx, fy + sy)
                for sx, sy in [(0, 1), (0, -1), (1, 0), (-1, 0)]
                if (fx + sx, fy + sy) != (cx, cy)
            ]
            if any(
                0 <= sx < w and 0 <= sy < h and new_grid[sy, sx] == 1
                for sx, sy in side_neighbors
            ):
                break
            new_grid[fy, fx] = 1
            cx, cy = fx, fy
            carved += 1
        else:
            break

    return new_grid


def dfs_exploration_with_backtracking(
    maze_grid: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
) -> str:
    """Generate DFS exploration log with backtracking (for solvable mazes).

    Returns thinking text with <|point|> tags showing exploration process.
    """
    h, w = maze_grid.shape
    visited = set()
    stack = [(start, [start])]
    thinking_lines = []

    while stack:
        (x, y), path = stack.pop()
        norm_x = normalize_coordinate(x, w)
        norm_y = normalize_coordinate(y, h)

        if (x, y) == end:
            thinking_lines.append(
                f"Reached destination at {format_point([(norm_x, norm_y)])}"
            )
            break

        if (x, y) in visited:
            continue
        visited.add((x, y))

        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze_grid[ny, nx] == 1 and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if not neighbors:
            thinking_lines.append(
                f"Dead end at {format_point([(norm_x, norm_y)])}, backtracking..."
            )
            continue

        for nx, ny in reversed(neighbors):
            stack.append(((nx, ny), path + [(nx, ny)]))
            thinking_lines.append(
                f"Move to {format_point([(normalize_coordinate(nx, w), normalize_coordinate(ny, h))])}"
            )

    return "\n".join(thinking_lines)


def dfs_exhaustive_search_proving_unreachable(
    maze_grid: np.ndarray,
    start: Tuple[int, int],
) -> str:
    """Generate exhaustive DFS log for unsolvable maze.

    Shows exploration of all reachable areas, then concludes no path exists.
    """
    h, w = maze_grid.shape
    visited = set()
    stack = [start]
    thinking_lines = []

    while stack:
        x, y = stack.pop()
        norm_x = normalize_coordinate(x, w)
        norm_y = normalize_coordinate(y, h)

        if (x, y) in visited:
            continue
        visited.add((x, y))

        thinking_lines.append(
            f"Exploring {format_point([(norm_x, norm_y)])}"
        )

        neighbors = []
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < w and 0 <= ny < h and maze_grid[ny, nx] == 1 and (nx, ny) not in visited:
                neighbors.append((nx, ny))

        if not neighbors:
            thinking_lines.append(
                f"Dead end at {format_point([(norm_x, norm_y)])}, backtracking..."
            )
        else:
            for nx, ny in reversed(neighbors):
                stack.append((nx, ny))

    thinking_lines.append(
        "After exploring all reachable areas, no path to the destination exists."
    )
    return "\n".join(thinking_lines)


def render_maze_image(
    maze_grid: np.ndarray,
    start: Tuple[int, int] | None = None,
    end: Tuple[int, int] | None = None,
    cell_size: int = 20,
) -> Image.Image:
    """Render maze as PIL Image."""
    h, w = maze_grid.shape
    img_h = h * cell_size
    img_w = w * cell_size
    img = Image.new("RGB", (img_w, img_h), "white")
    draw = ImageDraw.Draw(img)

    for y in range(h):
        for x in range(w):
            if maze_grid[y, x] == 0:
                draw.rectangle(
                    [x * cell_size, y * cell_size, (x + 1) * cell_size, (y + 1) * cell_size],
                    fill="#333333",
                )

    if start:
        sx, sy = start
        draw.ellipse(
            [sx * cell_size + 4, sy * cell_size + 4,
             (sx + 1) * cell_size - 4, (sy + 1) * cell_size - 4],
            fill="#00FF00",
        )
    if end:
        ex, ey = end
        draw.rectangle(
            [ex * cell_size + 4, ey * cell_size + 4,
             (ex + 1) * cell_size - 4, (ey + 1) * cell_size - 4],
            fill="#FF0000",
        )

    return img


def _build_maze_thinking_3step(grounding_log: str, solvable: bool) -> str:
    """Wrap maze DFS log in 3-step thinking protocol."""
    intent = (
        "I need to determine if there is a path from the green circle "
        "(start) to the red square (end) in the maze."
    )
    summarization = (
        "A path exists from start to end."
        if solvable else
        "No path to the destination exists."
    )
    return (
        f"Intent Analysis: {intent}\n"
        f"Grounding: {grounding_log}\n"
        f"Summarization: {summarization}"
    )


def _limit_grounding_log(grounding: str, max_lines: int = 25) -> str:
    """Limit DFS grounding log length while preserving structure.

    Keeps the first and last portions of the exploration with a summary
    line in between, so that:
    - The initial exploration direction is preserved
    - The final conclusion (e.g. "Reached destination") is kept
    - The 3-step thinking structure remains intact after wrapping
    """
    if max_lines < 2:
        max_lines = 2  # need at least 1 head + 1 tail line
    lines = grounding.split("\n")
    if len(lines) <= max_lines:
        return grounding

    keep = max_lines // 2
    skipped = len(lines) - 2 * keep
    head = lines[:keep]
    tail = lines[-keep:]
    summary = f"... (explored {skipped} more cells) ..."
    return "\n".join(head + [summary] + tail)


def generate_maze_dataset(
    n: int = 100000,
    seed: int = 42,
    cache_dir: str = "data/cache/maze",
    max_grounding_lines: int = 25,
    backtracking_trap_ratio: float = 0.5,
) -> List[Dict]:
    """Generate maze dataset with 50% solvable / 50% unsolvable and 3-step thinking.

    Args:
        backtracking_trap_ratio: Fraction of solvable mazes that get an explicit
            dead-end trap carved next to the solution path, forcing backtracking.

    Images are saved to disk and image_path is returned to avoid OOM.

    Returns list of dicts with keys:
        image: str (path to saved image)
        prompt: str
        reasoning: str (3-step thinking)
        answer: str
        maze_grid: np.ndarray
        task_type: str = "maze"
    """
    import os

    random.seed(seed)
    np.random.seed(seed)
    os.makedirs(cache_dir, exist_ok=True)
    data = []

    for idx in range(n):
        grid_size = random.randint(8, 12)
        maze_grid = generate_maze_grid(grid_size, grid_size)
        start, end = find_start_end(maze_grid)

        if start is None or end is None:
            continue

        if random.random() < MAZE_SOLVABLE_RATIO:
            # Solvable maze; optionally add a backtracking trap.
            solution = solve_maze_dfs(maze_grid, start, end)
            if solution and random.random() < backtracking_trap_ratio:
                maze_grid = add_backtracking_trap(maze_grid, solution, trap_length=random.randint(2, 4))
            grounding = dfs_exploration_with_backtracking(maze_grid, start, end)
            grounding = _limit_grounding_log(grounding, max_grounding_lines)
            thinking = _build_maze_thinking_3step(grounding, solvable=True)
            answer = r"\boxed{True}"
        else:
            # Unsolvable: break path in middle
            solution = solve_maze_dfs(maze_grid, start, end)
            if solution:
                maze_grid = break_path_middle(maze_grid, solution)
            grounding = dfs_exhaustive_search_proving_unreachable(maze_grid, start)
            grounding = _limit_grounding_log(grounding, max_grounding_lines)
            thinking = _build_maze_thinking_3step(grounding, solvable=False)
            answer = r"\boxed{False}"

        img = render_maze_image(maze_grid, start, end)
        img_path = os.path.join(cache_dir, f"maze_{seed}_{idx:06d}.png")
        img.save(img_path)

        data.append({
            "image": img_path,
            "prompt": "Is there a path from the green circle (start) to the red square (end)? Use <|point|> to show your exploration process step by step.",
            "reasoning": thinking,
            "answer": answer,
            "maze_grid": maze_grid,
            "task_type": "maze",
        })

    import logging
    logger = logging.getLogger(__name__)
    data = filter_verified_samples(data, logger=logger)
    return data
