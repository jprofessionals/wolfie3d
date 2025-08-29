#!/usr/bin/env python3
"""
Vibe Wolf (Python + PyOpenGL) — GL-renderer
-------------------------------------------
Denne varianten beholder logikken (kart, DDA-raycasting, input, sprites),
men tegner ALT med OpenGL (GPU). Vegger og sprites blir teksturerte quads,
og vi bruker depth-test i GPU for korrekt okklusjon (ingen CPU zbuffer).

Avhengigheter:
  - pygame >= 2.1 (for vindu/input)
  - PyOpenGL, PyOpenGL-accelerate
  - numpy

Kjør:
  python wolfie3d_gl.py

Taster:
  - WASD / piltaster: bevegelse
  - Q/E eller ← → : rotasjon
  - SPACE / venstre mus: skyte
  - ESC: avslutt
"""

from __future__ import annotations

import math
import sys
from typing import TYPE_CHECKING

import numpy as np
import pygame
from OpenGL import GL as gl

if TYPE_CHECKING:  # kun for typing hints
    from collections.abc import Sequence

# ---------- Konfig ----------
WIDTH, HEIGHT = 1024, 600
HALF_W, HALF_H = WIDTH // 2, HEIGHT // 2
FPS = 60

# Kamera/FOV
FOV = 66 * math.pi / 180.0
PLANE_LEN = math.tan(FOV / 2)

# Bevegelse
MOVE_SPEED = 3.0      # enheter/sek
ROT_SPEED = 2.0       # rad/sek
STRAFE_SPEED = 2.5

# Tekstur-størrelse brukt på GPU (proseduralt generert)
TEX_W = TEX_H = 256
# Små epsilon-verdier for å unngå sampling helt i ytterkant av teksturer
TEX_EPS = 0.0005

# Depth mapping (lineær til [0..1] for gl_FragDepth)
FAR_PLANE = 100.0

# Dør-konstanter
DOOR_TILE_ID = 9        # spesialverdi i MAP for dør-celle
DOOR_TEX_ID = 3         # bruker treteksturen visuelt
DOOR_ANIM_TIME = 0.2    # sekunder opp/ned
DOOR_AUTO_CLOSE = 2.0   # sekunder etter åpning

# Kart (0=tomt, >0=veggtype/tekstur-id)
MAP: list[list[int]] = [
    # 28x20 kart, dør ved (19,10) som åpner inn i ny seksjon (rom) til høyre
    # 0..19 = original, 20..27 = utvidelse
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,1,1,1,1,1,1,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1, 1,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3, 0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1, 1,0,1,0,0,0,0,1],
    [1,0,2,2,2,2,2,0,0,0,0,3,3,3,0,0,4,4,4,1, 1,0,1,1,1,1,1,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1, 1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1, 1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1, 1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1, 1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1, 1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1, 1,1,1,1,1,1,1,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1, 1,1,1,1,1,1,1,1],
]
MAP_W = len(MAP[0])
MAP_H = len(MAP)

# Dørposisjon (kobling til ny seksjon)
DOOR_X, DOOR_Y = 19, 10

# Startpos og retning
player_x = 3.5
player_y = 10.5
dir_x, dir_y = 1.0, 0.0
plane_x, plane_y = 0.0, PLANE_LEN

# ---------- Hjelpere ----------
def in_map(ix: int, iy: int) -> bool:
    return 0 <= ix < MAP_W and 0 <= iy < MAP_H

def is_wall(ix: int, iy: int) -> bool:
    if not in_map(ix, iy):
        return False
    cell = MAP[iy][ix]
    if cell == DOOR_TILE_ID:
        d = doors.get((ix, iy))
        # blokkér som vegg hvis ikke fullstendig åpen
        return not (d and d.anim >= 0.999)
    return cell > 0

def clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

# Sett over synlige/utforskede fliser for minimap
seen_tiles: set[tuple[int, int]] = set()

def compute_reachable() -> set[tuple[int, int]]:
    """Flood fill fra spillerposisjon gjennom åpne celler (0) og HELT åpne dører.
    Returnerer sett av passérbare fliser (ikke vegger)."""
    sx, sy = int(player_x), int(player_y)
    if not in_map(sx, sy):
        return set()
    q: list[tuple[int, int]] = [(sx, sy)]
    visited: set[tuple[int, int]] = {(sx, sy)}
    def passable(ix: int, iy: int) -> bool:
        if not in_map(ix, iy):
            return False
        cell = MAP[iy][ix]
        if cell == 0:
            return True
        if cell == DOOR_TILE_ID:
            d = doors.get((ix, iy))
            return bool(d and d.anim >= 0.999)
        return False
    while q:
        x, y = q.pop(0)
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            nx, ny = x+dx, y+dy
            if (nx, ny) in visited:
                continue
            if passable(nx, ny):
                visited.add((nx, ny))
                q.append((nx, ny))
    return visited

# ---------- Prosjektil ----------
class Bullet:
    def __init__(self, x: float, y: float, vx: float, vy: float) -> None:
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.alive = True
        self.age = 0.0
        self.height_param = 0.2  # 0..~0.65 (stiger visuelt)

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        nx = self.x + self.vx * dt
        ny = self.y + self.vy * dt
        if is_wall(int(nx), int(ny)):
            self.alive = False
            return
        self.x, self.y = nx, ny
        self.age += dt
        self.height_param = min(0.65, self.height_param + 0.35 * dt)

# ---------- Fiende ----------
class Enemy:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.alive = True
        self.radius = 0.35     # kollisjon/hitbox i kart-enheter
        self.speed = 1.4       # enheter/sek (enkel jakt)
        self.height_param = 0.5  # hvor høyt sprite sentreres i skjerm

    def _try_move(self, nx: float, ny: float) -> None:
        # enkel vegg-kollisjon (sirkulær hitbox mot grid)
        # prøv X:
        if not is_wall(int(nx), int(self.y)):
            self.x = nx
        # prøv Y:
        if not is_wall(int(self.x), int(ny)):
            self.y = ny

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        # enkel "chase": gå mot spilleren, stopp om rett foran vegg
        dx = player_x - self.x
        dy = player_y - self.y
        dist = math.hypot(dx, dy) + 1e-9
        # ikke gå helt oppå spilleren
        if dist > 0.75:
            ux, uy = dx / dist, dy / dist
            step = self.speed * dt
            self._try_move(self.x + ux * step, self.y + uy * step)


# ---------- Dør ----------
class Door:
    def __init__(self, x: int, y: int, tex_id: int = DOOR_TEX_ID) -> None:
        self.x = x
        self.y = y
        self.tex_id = tex_id
        self.state = "closed"  # closed/opening/open/closing
        self.anim = 0.0         # 0.0 = lukket (full høyde), 1.0 = åpen (skjult)
        self.timer = -1.0       # auto-close nedtelling når open
        self.pending_close = False

    def is_near_player(self) -> bool:
        return int(player_x) == self.x and int(player_y) == self.y

    def any_actor_in_tile(self, enemies: list['Enemy']) -> bool:
        if int(player_x) == self.x and int(player_y) == self.y:
            return True
        for e in enemies:
            if not e.alive:
                continue
            if int(e.x) == self.x and int(e.y) == self.y:
                return True
        return False

    def request_open(self) -> None:
        if self.state in ("closed", "closing"):
            self.state = "opening"
            self.pending_close = False

    def update(self, dt: float, enemies: list['Enemy']) -> None:
        # anim
        if self.state == "opening":
            self.anim += dt / DOOR_ANIM_TIME
            if self.anim >= 1.0:
                self.anim = 1.0
                self.state = "open"
                self.timer = DOOR_AUTO_CLOSE
        elif self.state == "closing":
            self.anim -= dt / DOOR_ANIM_TIME
            if self.anim <= 0.0:
                self.anim = 0.0
                self.state = "closed"
                self.timer = -1.0

        # auto-close
        if self.state == "open":
            if self.timer > 0.0:
                self.timer -= dt
            if self.timer <= 0.0:
                # prøv å lukke hvis mulig
                if not self.any_actor_in_tile(enemies):
                    self.state = "closing"
                else:
                    # blokkert – forsøk igjen når cellen blir ledig
                    self.pending_close = True
                    self.timer = -1.0

        # hvis ventende lukking og cellen er ledig – lukk straks
        if self.pending_close and self.state == "open" and not self.any_actor_in_tile(enemies):
            self.state = "closing"
            self.pending_close = False


# Samling av dører i kartet
doors: dict[tuple[int, int], Door] = {}


# ---------- Prosedural tekstur (pygame.Surface) ----------
def make_brick_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    surf.fill((150, 40, 40))
    mortar = (200, 200, 200)
    brick_h = TEX_H // 4
    brick_w = TEX_W // 4
    for row in range(0, TEX_H, brick_h):
        offset = 0 if (row // brick_h) % 2 == 0 else brick_w // 2
        for col in range(0, TEX_W, brick_w):
            rect = pygame.Rect((col + offset) % TEX_W, row, brick_w - 1, brick_h - 1)
            pygame.draw.rect(surf, (165, 52, 52), rect)
    for y in range(0, TEX_H, brick_h):
        pygame.draw.line(surf, mortar, (0, y), (TEX_W, y))
    for x in range(0, TEX_W, brick_w):
        pygame.draw.line(surf, mortar, (x, 0), (x, TEX_H))
    return surf

def make_stone_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    base = (110, 110, 120)
    surf.fill(base)
    for y in range(TEX_H):
        for x in range(TEX_W):
            if ((x * 13 + y * 7) ^ (x * 3 - y * 5)) & 15 == 0:
                c = 90 + ((x * y) % 40)
                surf.set_at((x, y), (c, c, c))
    for i in range(5):
        pygame.draw.line(surf, (80, 80, 85), (i*12, 0), (TEX_W-1, TEX_H-1 - i*6), 1)
    return surf

def make_wood_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H))
    for y in range(TEX_H):
        for x in range(TEX_W):
            v = int(120 + 40 * math.sin((x + y*0.5) * 0.12) + 20 * math.sin(y * 0.3))
            v = max(60, min(200, v))
            surf.set_at((x, y), (140, v, 60))
    for x in range(0, TEX_W, TEX_W // 4):
        pygame.draw.line(surf, (90, 60, 30), (x, 0), (x, TEX_H))
    return surf

def make_metal_texture() -> pygame.Surface:
    surf = pygame.Surface((TEX_W, TEX_H), pygame.SRCALPHA)
    base = (140, 145, 150, 255)
    surf.fill(base)
    for y in range(8, TEX_H, 16):
        for x in range(8, TEX_W, 16):
            pygame.draw.circle(surf, (90, 95, 100, 255), (x, y), 2)
    for y in range(TEX_H):
        shade = 130 + (y % 8) * 2
        pygame.draw.line(surf, (shade, shade, shade+5, 255), (0, y), (TEX_W, y), 1)
    return surf

def make_bullet_texture() -> pygame.Surface:
    surf = pygame.Surface((32, 32), pygame.SRCALPHA)
    pygame.draw.circle(surf, (255, 240, 150, 220), (16, 16), 8)
    pygame.draw.circle(surf, (255, 255, 255, 255), (13, 13), 3)
    return surf

# ---------- OpenGL utils ----------
VS_SRC = """
#version 330 core
layout (location = 0) in vec2 in_pos;    // NDC -1..1
layout (location = 1) in vec2 in_uv;
layout (location = 2) in vec3 in_col;    // per-vertex farge (for dimming/overlay)
layout (location = 3) in float in_depth; // 0..1 depth (0 nær, 1 fjern)

out vec2 v_uv;
out vec3 v_col;
out float v_depth;

void main() {
    v_uv = in_uv;
    v_col = in_col;
    v_depth = in_depth;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""

FS_SRC = """
#version 330 core
in vec2 v_uv;
in vec3 v_col;
in float v_depth;

out vec4 fragColor;

uniform sampler2D uTexture;
uniform bool uUseTexture;

void main() {
    vec4 base = vec4(1.0);
    if (uUseTexture) {
        base = texture(uTexture, v_uv);
        if (base.a < 0.01) discard; // alpha for sprites
    }
    vec3 rgb = base.rgb * v_col;
    fragColor = vec4(rgb, base.a);
    // Skriv eksplisitt dybde (lineær i [0..1])
    gl_FragDepth = clamp(v_depth, 0.0, 1.0);
}
"""

def compile_shader(src: str, stage: int) -> int:
    sid = gl.glCreateShader(stage)
    gl.glShaderSource(sid, src)
    gl.glCompileShader(sid)
    status = gl.glGetShaderiv(sid, gl.GL_COMPILE_STATUS)
    if status != gl.GL_TRUE:
        log = gl.glGetShaderInfoLog(sid).decode()
        raise RuntimeError(f"Shader compile error:\n{log}")
    return sid

def make_program(vs_src: str, fs_src: str) -> int:
    vs = compile_shader(vs_src, gl.GL_VERTEX_SHADER)
    fs = compile_shader(fs_src, gl.GL_FRAGMENT_SHADER)
    prog = gl.glCreateProgram()
    gl.glAttachShader(prog, vs)
    gl.glAttachShader(prog, fs)
    gl.glLinkProgram(prog)
    ok = gl.glGetProgramiv(prog, gl.GL_LINK_STATUS)
    gl.glDeleteShader(vs)
    gl.glDeleteShader(fs)
    if ok != gl.GL_TRUE:
        log = gl.glGetProgramInfoLog(prog).decode()
        raise RuntimeError(f"Program link error:\n{log}")
    return prog

def surface_to_texture(surf: pygame.Surface, *, wrap: str = "repeat", filter_mode: str | None = None) -> int:
    """Laster pygame.Surface til GL_TEXTURE_2D (RGBA8). Returnerer texture id.
    wrap: 'repeat' for vegger (tile), 'clamp' for sprites/overlays.
    """
    data = pygame.image.tostring(surf.convert_alpha(), "RGBA", True)
    w, h = surf.get_width(), surf.get_height()
    tid = gl.glGenTextures(1)
    gl.glBindTexture(gl.GL_TEXTURE_2D, tid)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data)
    # Velg filtrering: for vegger gir NEAREST færre smear-artefakter ved 1px striper
    if filter_mode is None:
        filter_mode = "nearest" if wrap == "repeat" else "linear"
    if filter_mode == "nearest":
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
    else:
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    if wrap == "clamp":
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
    else:
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tid

def make_white_texture() -> int:
    surf = pygame.Surface((1, 1), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 255))
    return surface_to_texture(surf, wrap="clamp", filter_mode="linear")

def make_enemy_texture() -> pygame.Surface:
    s = pygame.Surface((256, 256), pygame.SRCALPHA)
    # kropp
    pygame.draw.rect(s, (60, 60, 70, 255), (100, 80, 56, 120), border_radius=6)
    # hode
    pygame.draw.circle(s, (220, 200, 180, 255), (128, 70), 26)
    # hjelm-ish
    pygame.draw.arc(s, (40, 40, 50, 255), (92, 40, 72, 40), 3.14, 0, 6)
    # “arm”
    pygame.draw.rect(s, (60, 60, 70, 255), (86, 110, 24, 16))
    pygame.draw.rect(s, (60, 60, 70, 255), (146, 110, 24, 16))
    return s


# ---------- GL Renderer state ----------
from pathlib import Path
import os
import pygame
from OpenGL import GL as gl

# ---------- GL Renderer state ----------
class GLRenderer:
    def __init__(self) -> None:
        # Shader program
        self.prog = make_program(VS_SRC, FS_SRC)
        gl.glUseProgram(self.prog)
        self.uni_tex = gl.glGetUniformLocation(self.prog, "uTexture")
        self.uni_use_tex = gl.glGetUniformLocation(self.prog, "uUseTexture")
        gl.glUniform1i(self.uni_tex, 0)

        # VAO/VBO (dynamisk buffer per draw)
        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)

        stride = 8 * 4  # 8 float32 per vertex
        # in_pos (loc 0): 2 floats
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        # in_uv (loc 1): 2 floats
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(2 * 4))
        # in_col (loc 2): 3 floats
        gl.glEnableVertexAttribArray(2)
        gl.glVertexAttribPointer(2, 3, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(4 * 4))
        # in_depth (loc 3): 1 float
        gl.glEnableVertexAttribArray(3)
        gl.glVertexAttribPointer(3, 1, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(7 * 4))

        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

        # Teksturer
        self.white_tex = make_white_texture()
        self.textures: dict[int, int] = {}  # tex_id -> GL texture

        # Last fra assets hvis tilgjengelig, ellers fall tilbake til proseduralt
        self.load_textures()

        # GL state
        gl.glEnable(gl.GL_BLEND)
        gl.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glDepthFunc(gl.GL_LEQUAL)

    # ---------- teksturhjelpere ----------
    @staticmethod
    def _scale_if_needed(surf: pygame.Surface, size: int = 512) -> pygame.Surface:
        if surf.get_width() != size or surf.get_height() != size:
            surf = pygame.transform.smoothscale(surf, (size, size))
        return surf

    def _load_texture_file(self, path: str, size: int = 512) -> int:
        surf = pygame.image.load(path).convert_alpha()
        surf = self._scale_if_needed(surf, size)
        return surface_to_texture(surf, wrap="repeat", filter_mode="nearest")

    # ---------- offentlig laster ----------

    def _resolve_textures_base(self) -> Path:
        """
        Finn korrekt assets/textures-katalog robust, uavhengig av hvor vi kjører fra.
        Prøver i rekkefølge:
          - <her>/assets/textures
          - <her>/../assets/textures
          - <her>/../../assets/textures      <-- typisk når koden ligger i src/wolfie3d
          - <cwd>/assets/textures
        """
        here = Path(__file__).resolve().parent
        candidates = [
            here / "assets" / "textures",
            here.parent / "assets" / "textures",
            here.parent.parent / "assets" / "textures",
            Path.cwd() / "assets" / "textures",
        ]
        print("\n[GLRenderer] Prøver å finne assets/textures på disse stedene:")
        for c in candidates:
            print("  -", c)
            if c.exists():
                print("[GLRenderer] FANT:", c)
                return c

        raise FileNotFoundError(
            "Fant ikke assets/textures i noen av kandidatkatalogene over. "
            "Opprett assets/textures på prosjektnivå (samme nivå som src) eller justér stien."
        )

    def load_textures(self) -> None:
        """
        Debug-variant som bruker korrekt prosjekt-rot og feiler høyt hvis filer mangler.
        Forventer: bricks.png, stone.png, wood.png, metal.png i assets/textures/.
        """
        base = self._resolve_textures_base()
        print(f"[GLRenderer] pygame extended image support: {pygame.image.get_extended()}")
        print(f"[GLRenderer] Innhold i {base}: {[p.name for p in base.glob('*')]}")

        files = {
            1: base / "bricks.png",
            2: base / "stone.png",
            3: base / "wood.png",
            4: base / "metal.png",
        }
        missing = [p for p in files.values() if not p.exists()]
        if missing:
            print("[GLRenderer] MANGEL: følgende filer finnes ikke:")
            for m in missing:
                print("  -", m)
            raise FileNotFoundError(
                "Manglende teksturer. Sørg for at filene ligger i assets/textures/")

        def _load(path: Path, size: int = 512) -> int:
            print(f"[GLRenderer] Laster: {path}")
            surf = pygame.image.load(str(path)).convert_alpha()
            if surf.get_width() != size or surf.get_height() != size:
                print(
                    f"[GLRenderer]  - rescale {surf.get_width()}x{surf.get_height()} -> {size}x{size}")
                surf = pygame.transform.smoothscale(surf, (size, size))
            tex_id = surface_to_texture(surf, wrap="repeat")
            print(f"[GLRenderer]  - OK (GL tex id {tex_id})")
            return tex_id

        self.textures[1] = _load(files[1], 512)
        self.textures[2] = _load(files[2], 512)
        self.textures[3] = _load(files[3], 512)
        self.textures[4] = _load(files[4], 512)

        # Sprite (kule) – behold prosedyre
        # bullet sprite har alpha: bruk clamp for å unngå fringe/bleed
        self.textures[99] = surface_to_texture(make_bullet_texture(), wrap="clamp")

        # Enemy sprite (ID 200): prøv fil, ellers prosedyral placeholder
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            enemy_path = sprites_dir / "enemy.png"
            print(f"[GLRenderer] Leter etter enemy sprite i: {enemy_path}")
            if enemy_path.exists():
                self.textures[200] = self._load_texture_file(enemy_path, 512)
                print(f"[GLRenderer] Enemy OK (GL tex id {self.textures[200]})")
            else:
                # fallback – prosedural fiende
                self.textures[200] = surface_to_texture(make_enemy_texture(), wrap="clamp")
                print("[GLRenderer] Enemy: bruker prosedural sprite")
        except Exception as ex:
            print(f"[GLRenderer] Enemy: FEIL ved lasting ({ex}), bruker prosedural")
            self.textures[200] = surface_to_texture(make_enemy_texture(), wrap="clamp")

        print("[GLRenderer] Teksturer lastet.\n")

    # ---------- draw ----------
    def draw_arrays(self, verts: np.ndarray, texture: int, use_tex: bool, *, alpha: bool = True) -> None:
        if verts.size == 0:
            return
        gl.glUseProgram(self.prog)
        gl.glUniform1i(self.uni_use_tex, 1 if use_tex else 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture if use_tex else self.white_tex)

        # For vegger/opaque overlay deaktiver blending for å unngå kant-artefakter
        if alpha:
            gl.glEnable(gl.GL_BLEND)
        else:
            gl.glDisable(gl.GL_BLEND)

        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, verts.nbytes, verts, gl.GL_DYNAMIC_DRAW)
        count = verts.shape[0]
        gl.glDrawArrays(gl.GL_TRIANGLES, 0, count)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
        gl.glBindVertexArray(0)

# ---------- Raycasting + bygg GL-verts ----------
def column_ndc(x: int) -> tuple[float, float]:
    """Returnerer venstre/høyre NDC-X for en 1-px bred skjermkolonne."""
    x_left = (2.0 * x) / WIDTH - 1.0
    x_right = (2.0 * (x + 1)) / WIDTH - 1.0
    # Unngå å lande eksakt på -1/1 for å forhindre rasteriseringsartefakter på vinduskant
    edge_eps = 1.0 / float(WIDTH)  # ~1px in NDC space
    if x == 0:
        x_left += edge_eps * 0.25
    if x == WIDTH - 1:
        x_right -= edge_eps * 0.25
    return x_left, x_right

def y_ndc(y_pix: int) -> float:
    """Konverter skjerm-Y (0 top) til NDC-Y (1 top, -1 bunn)."""
    return 1.0 - 2.0 * (y_pix / float(HEIGHT))

def dim_for_side(side: int) -> float:
    # dim litt på sidevegger (liknende BLEND_MULT tidligere)
    return 0.78 if side == 1 else 1.0

def cast_and_build_wall_batches() -> dict[int, list[float]]:
    # Ny implementasjon: lagrer bakvegg og dør uavhengig per kolonne, for korrekt perspektiv
    batches_new: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: [], 0: []}
    for x in range(WIDTH):
        camera_x = 2.0 * x / WIDTH - 1.0
        ray_dir_x = dir_x + plane_x * camera_x
        ray_dir_y = dir_y + plane_y * camera_x

        map_x = int(player_x)
        map_y = int(player_y)

        delta_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e30
        delta_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e30
        if ray_dir_x < 0:
            step_x = -1
            side_x = (player_x - map_x) * delta_x
        else:
            step_x = 1
            side_x = (map_x + 1.0 - player_x) * delta_x
        if ray_dir_y < 0:
            step_y = -1
            side_y = (player_y - map_y) * delta_y
        else:
            step_y = 1
            side_y = (map_y + 1.0 - player_y) * delta_y

        door_hit = None  # {'dist','side','u','anim'}
        wall_hit = None  # {'dist','side','u','tex'}

        for _ in range(256):
            if side_x < side_y:
                side_x += delta_x
                map_x += step_x
                side = 0
            else:
                side_y += delta_y
                map_y += step_y
                side = 1
            if not in_map(map_x, map_y):
                break
            cell = MAP[map_y][map_x]
            if cell <= 0:
                continue
            # avstand og u
            if side == 0:
                dist = (map_x - player_x + (1 - step_x) / 2.0) / (ray_dir_x if ray_dir_x != 0 else 1e-9)
                wx = player_y + dist * ray_dir_y
            else:
                dist = (map_y - player_y + (1 - step_y) / 2.0) / (ray_dir_y if ray_dir_y != 0 else 1e-9)
                wx = player_x + dist * ray_dir_x
            wx -= math.floor(wx)
            u = wx
            if (side == 0 and ray_dir_x > 0) or (side == 1 and ray_dir_y < 0):
                u = 1.0 - u
            if u <= TEX_EPS:
                u = TEX_EPS
            elif u >= 1.0 - TEX_EPS:
                u = 1.0 - TEX_EPS

            if cell == DOOR_TILE_ID:
                d = doors.get((map_x, map_y))
                p = d.anim if d else 0.0
                if p >= 0.999:
                    continue
                if door_hit is None or dist < door_hit['dist']:
                    door_hit = {
                        'dist': dist,
                        'side': side,
                        'u': u,
                        'anim': max(0.0, min(1.0, p)),
                        'state': (d.state if d else 'opening'),
                    }
                continue
            else:
                wall_hit = {'dist': dist, 'side': side, 'u': u, 'tex': cell}
                break

        x_left, x_right = column_ndc(x)
        # Bakvegg
        if wall_hit is not None:
            dist = wall_hit['dist']
            sidew = wall_hit['side']
            uw = wall_hit['u']
            tex = wall_hit['tex']
            lh = int(HEIGHT / (dist + 1e-6))
            rs0 = -lh / 2.0 + HALF_H
            re0 = rs0 + lh
            ds = max(0, int(rs0))
            de = min(HEIGHT - 1, int(re0))
            top = y_ndc(ds)
            bot = y_ndc(de)
            v0 = (ds - rs0) / max(1.0, float(lh))
            v1 = (de - rs0) / max(1.0, float(lh))
            v0 = min(1.0 - TEX_EPS, max(TEX_EPS, v0))
            v1 = min(1.0 - TEX_EPS, max(TEX_EPS, v1))
            c = dim_for_side(sidew)
            r = g = b = c
            depth = clamp01(dist / FAR_PLANE)
            arr = [
                x_left,  top, uw, v0, r, g, b, depth,
                x_left,  bot, uw, v1, r, g, b, depth,
                x_right, top, uw, v0, r, g, b, depth,
                x_right, top, uw, v0, r, g, b, depth,
                x_left,  bot, uw, v1, r, g, b, depth,
                x_right, bot, uw, v1, r, g, b, depth,
            ]
            batches_new.setdefault(tex, []).extend(arr)

        # Dørpanel
        if door_hit is not None:
            dist = door_hit['dist']
            sided = door_hit['side']
            ud = door_hit['u']
            p = door_hit['anim']
            st = door_hit.get('state', 'opening')
            lh = int(HEIGHT / (dist + 1e-6))
            rs0 = -lh / 2.0 + HALF_H
            re0 = rs0 + lh
            if st == 'closing':
                # Slide up: keep bottom fixed, erase from top as p decreases
                rs = max(rs0, re0 - (1.0 - p) * lh)
                re = re0
            else:
                # Opening: slide down, top moves down, bottom fixed
                rs = min(re0, rs0 + p * lh)
                re = re0
            ds = max(0, int(rs))
            de = min(HEIGHT - 1, int(re))
            if ds < de:
                top = y_ndc(ds)
                bot = y_ndc(de)
                # Texture slides with the panel: use a global offset of -p and allow wrapping.
                # This makes the content translate downward during opening and upward during closing.
                tex_off = -p
                v0 = ((ds - rs0) / max(1.0, float(lh))) + tex_off
                v1 = ((de - rs0) / max(1.0, float(lh))) + tex_off
                c = dim_for_side(sided)
                r = g = b = c
                depth = clamp01(dist / FAR_PLANE)
                arr = [
                    x_left,  top, ud, v0, r, g, b, depth,
                    x_left,  bot, ud, v1, r, g, b, depth,
                    x_right, top, ud, v0, r, g, b, depth,
                    x_right, top, ud, v0, r, g, b, depth,
                    x_left,  bot, ud, v1, r, g, b, depth,
                    x_right, bot, ud, v1, r, g, b, depth,
                ]
                batches_new.setdefault(DOOR_TEX_ID, []).extend(arr)

    return batches_new
    batches: dict[int, list[float]] = {1: [], 2: [], 3: [], 4: []}
    for x in range(WIDTH):
        # Raydir
        camera_x = 2.0 * x / WIDTH - 1.0
        ray_dir_x = dir_x + plane_x * camera_x
        ray_dir_y = dir_y + plane_y * camera_x
        map_x = int(player_x)
        map_y = int(player_y)

        delta_dist_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e30
        delta_dist_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e30

        if ray_dir_x < 0:
            step_x = -1
            side_dist_x = (player_x - map_x) * delta_dist_x
        else:
            step_x = 1
            side_dist_x = (map_x + 1.0 - player_x) * delta_dist_x
        if ray_dir_y < 0:
            step_y = -1
            side_dist_y = (player_y - map_y) * delta_dist_y
        else:
            step_y = 1
            side_dist_y = (map_y + 1.0 - player_y) * delta_dist_y

        hit = 0
        side = 0
        tex_id = 1
        door_progress = 0.0
        door_state = None
        # Data for "beyond" treff når døren er delvis åpen
        beyond_hit = None  # tuple(perp_dist, u, side, tex_id)
        while hit == 0:
            if side_dist_x < side_dist_y:
                side_dist_x += delta_dist_x
                map_x += step_x
                side = 0
            else:
                side_dist_y += delta_dist_y
                map_y += step_y
                side = 1
            if not in_map(map_x, map_y):
                hit = 1
                tex_id = 1
                break
            cell = MAP[map_y][map_x]
            if cell > 0:
                # Dør: hvis fullstendig åpen – fortsett DDA gjennom cellen
                if cell == DOOR_TILE_ID:
                    d = doors.get((map_x, map_y))
                    if d and d.anim >= 0.999:
                        continue  # åpen – ingen kollisjon
                    # ellers treff: rendres med tretekstur og klipp etter anim
                    hit = 1
                    tex_id = DOOR_TEX_ID
                    door_progress = (d.anim if d else 0.0)
                    door_state = (d.state if d else None)
                else:
                    hit = 1
                    tex_id = cell

        if side == 0:
            perp_wall_dist = (map_x - player_x + (1 - step_x) / 2.0) / (ray_dir_x if ray_dir_x != 0 else 1e-9)
            wall_x = player_y + perp_wall_dist * ray_dir_y
        else:
            perp_wall_dist = (map_y - player_y + (1 - step_y) / 2.0) / (ray_dir_y if ray_dir_y != 0 else 1e-9)
            wall_x = player_x + perp_wall_dist * ray_dir_x

        wall_x -= math.floor(wall_x)
        # u-koordinat (kontinuerlig) + flip for samsvar med klassisk raycaster
        u = wall_x
        if (side == 0 and ray_dir_x > 0) or (side == 1 and ray_dir_y < 0):
            u = 1.0 - u
        # Unngå sampling helt i kantene (0/1) for å redusere "dragging" pga. lineær filtrering
        if u <= TEX_EPS:
            u = TEX_EPS
        elif u >= 1.0 - TEX_EPS:
            u = 1.0 - TEX_EPS

        # skjermhøyde på vegg
        line_height = int(HEIGHT / (perp_wall_dist + 1e-6))
        raw_start0 = -line_height / 2.0 + HALF_H
        raw_end0 = raw_start0 + line_height

        # Dør: synker ned ved åpning, vokser nedefra ved lukking
        raw_start = raw_start0
        raw_end = raw_end0
        if door_state is not None:
            p = max(0.0, min(1.0, door_progress))
            if door_state in ("opening", "open"):
                # toppen flyttes nedover med p
                raw_start = min(raw_end0, raw_start0 + p * line_height)
                raw_end = raw_end0
            elif door_state == "closing":
                # toppen flyttes oppover fra bunnen
                visible_h = (1.0 - p) * line_height
                raw_start = max(raw_start0, raw_end0 - visible_h)
                raw_end = raw_end0
        # Klipp til skjerm
        draw_start = max(0, int(raw_start))
        draw_end = min(HEIGHT - 1, int(raw_end))

        # NDC koordinater for 1-px bred stripe
        x_left, x_right = column_ndc(x)
        top_ndc = y_ndc(draw_start)
        bot_ndc = y_ndc(draw_end)

        # v-koordinater: måles alltid relativt til full høyde (raw_start0)
        v0 = (draw_start - raw_start0) / max(1.0, float(line_height))
        v1 = (draw_end - raw_start0) / max(1.0, float(line_height))
        # klem til [0,1] og unngå helt i kantene
        v0 = min(1.0 - TEX_EPS, max(TEX_EPS, v0))
        v1 = min(1.0 - TEX_EPS, max(TEX_EPS, v1))

        # Farge-dim (samme på hele kolonnen)
        c = dim_for_side(side)
        r = g = b = c

        # Depth som lineær [0..1] (0 = nærmest)
        depth = clamp01(perp_wall_dist / FAR_PLANE)

        # 2 triangler (6 vertikser). Vertex-layout:
        # [x, y, u, v, r, g, b, depth]
        v = [
            # tri 1
            x_left,  top_ndc, u, v0, r, g, b, depth,
            x_left,  bot_ndc, u, v1, r, g, b, depth,
            x_right, top_ndc, u, v0, r, g, b, depth,
            # tri 2
            x_right, top_ndc, u, v0, r, g, b, depth,
            x_left,  bot_ndc, u, v1, r, g, b, depth,
            x_right, bot_ndc, u, v1, r, g, b, depth,
        ]
        batches.setdefault(tex_id, []).extend(v)

        # Under animasjon – tegn vegg/rom bak i åpningen (se gjennom), samt en tynn toppstripe
        # i gulvfarge for dørens overside ved åpning/lukking.
        if door_state in ("opening", "closing") and (door_progress > 0.0) and (door_progress < 0.999):
            # Fortsett DDA rett bak døren
            t_offset = max(0.0, perp_wall_dist) + 1e-4
            ox = player_x + ray_dir_x * t_offset
            oy = player_y + ray_dir_y * t_offset
            mx = int(ox)
            my = int(oy)
            ddx = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e30
            ddy = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e30
            if ray_dir_x < 0:
                stepx2 = -1
                sdx2 = (ox - mx) * ddx
            else:
                stepx2 = 1
                sdx2 = (mx + 1.0 - ox) * ddx
            if ray_dir_y < 0:
                stepy2 = -1
                sdy2 = (oy - my) * ddy
            else:
                stepy2 = 1
                sdy2 = (my + 1.0 - oy) * ddy
            side2 = 0
            hit2 = False
            tex2 = 1
            for _ in range(128):
                if sdx2 < sdy2:
                    sdx2 += ddx
                    mx += stepx2
                    side2 = 0
                else:
                    sdy2 += ddy
                    my += stepy2
                    side2 = 1
                if not in_map(mx, my):
                    break
                cell2 = MAP[my][mx]
                if cell2 > 0:
                    if cell2 == DOOR_TILE_ID:
                        d2 = doors.get((mx, my))
                        if d2 and d2.anim >= 0.999:
                            continue
                        tex2 = DOOR_TEX_ID
                        hit2 = True
                    else:
                        tex2 = cell2
                        hit2 = True
                if hit2:
                    if side2 == 0:
                        perp2 = (mx - ox + (1 - stepx2) / 2.0) / (ray_dir_x if ray_dir_x != 0 else 1e-9)
                        wallx2 = oy + perp2 * ray_dir_y
                    else:
                        perp2 = (my - oy + (1 - stepy2) / 2.0) / (ray_dir_y if ray_dir_y != 0 else 1e-9)
                        wallx2 = ox + perp2 * ray_dir_x
                    wallx2 -= math.floor(wallx2)
                    u2 = wallx2
                    if (side2 == 0 and ray_dir_x > 0) or (side2 == 1 and ray_dir_y < 0):
                        u2 = 1.0 - u2
                    u2 = min(1.0 - TEX_EPS, max(TEX_EPS, u2))

                    line2 = int(HEIGHT / (perp2 + 1e-6))
                    raw_s20 = -line2 / 2.0 + HALF_H
                    raw_e20 = raw_s20 + line2
                    dstart2 = max(0, int(raw_s20))
                    dend2 = min(HEIGHT - 1, int(raw_e20))

                    # Tegn HELE bakvegg-kolonnen; dørpanelet vil occludere korrekt via depth
                    top2 = y_ndc(dstart2)
                    bot2 = y_ndc(dend2)
                    c2 = dim_for_side(side2)
                    r2 = g2 = b2 = c2
                    depth2 = clamp01(perp2 / FAR_PLANE)
                    v0_2 = (dstart2 - raw_s20) / max(1.0, float(line2))
                    v1_2 = (dend2 - raw_s20) / max(1.0, float(line2))
                    v0_2 = min(1.0 - TEX_EPS, max(TEX_EPS, v0_2))
                    v1_2 = min(1.0 - TEX_EPS, max(TEX_EPS, v1_2))
                    vv = [
                        x_left,  top2, u2, v0_2, r2, g2, b2, depth2,
                        x_left,  bot2, u2, v1_2, r2, g2, b2, depth2,
                        x_right, top2, u2, v0_2, r2, g2, b2, depth2,
                        x_right, top2, u2, v0_2, r2, g2, b2, depth2,
                        x_left,  bot2, u2, v1_2, r2, g2, b2, depth2,
                        x_right, bot2, u2, v1_2, r2, g2, b2, depth2,
                    ]
                    batches.setdefault(tex2 if tex2 in (1,2,3,4) else DOOR_TEX_ID, []).extend(vv)
                    break

            # Toppflate (tynn stripe) i gulvfarge ved dørens nåværende topp
            if draw_start > 0:
                top_strip = 2
                y0p = max(0, draw_start - top_strip)
                y1p = draw_start
                x0 = x_left
                x1 = x_right
                y0 = y_ndc(y0p)
                y1 = y_ndc(y1p)
                r3, g3, b3 = (35/255.0, 35/255.0, 35/255.0)
                d3 = depth
                vv2 = [
                    x0, y0, 0.0, 0.0, r3, g3, b3, d3,
                    x0, y1, 0.0, 1.0, r3, g3, b3, d3,
                    x1, y0, 1.0, 0.0, r3, g3, b3, d3,
                    x1, y0, 1.0, 0.0, r3, g3, b3, d3,
                    x0, y1, 0.0, 1.0, r3, g3, b3, d3,
                    x1, y1, 1.0, 1.0, r3, g3, b3, d3,
                ]
                batches.setdefault(0, []).extend(vv2)
    return batches

def build_fullscreen_background() -> np.ndarray:
    """To store quads (himmel/gulv), farget med vertex-color, tegnes uten tekstur."""
    # Himmel (øverst halvdel)
    sky_col = (40/255.0, 60/255.0, 90/255.0)
    floor_col = (35/255.0, 35/255.0, 35/255.0)
    verts: list[float] = []

    # Quad helper
    def add_quad(x0, y0, x1, y1, col):
        r, g, b = col
        depth = 1.0  # lengst bak
        # u,v er 0 (vi bruker hvit 1x1 tekstur)
        verts.extend([
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,

            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ])

    # Koordinater i NDC
    add_quad(-1.0,  1.0,  1.0,  0.0, sky_col)   # øvre halvdel
    add_quad(-1.0,  0.0,  1.0, -1.0, floor_col) # nedre halvdel
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_sprites_batch(bullets: list[Bullet]) -> np.ndarray:
    """Bygger ett quad per kule i skjermen (billboard), med depth."""
    verts: list[float] = []

    for b in bullets:
        # Transform til kamera-rom
        spr_x = b.x - player_x
        spr_y = b.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue  # bak kamera

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))

        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk

        # vertikal offset: "stiger"
        v_shift = int((0.5 - b.height_param) * sprite_h)
        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y   = min(HEIGHT - 1, draw_start_y + sprite_h)
        # horisontal
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x   = draw_start_x + sprite_w

        # Klipp utenfor skjerm
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue
        draw_start_x = max(0, draw_start_x)
        draw_end_x   = min(WIDTH - 1, draw_end_x)

        # Konverter til NDC
        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = y_ndc(draw_start_y)
        y1 = y_ndc(draw_end_y)

        # Depth (basert på trans_y)
        depth = clamp01(trans_y / FAR_PLANE)

        r = g = bcol = 1.0  # ingen ekstra farge-dim
        # u,v: full tekstur
        u0, v0 = 0.0, 0.0
        u1, v1 = 1.0, 1.0

        verts.extend([
            x0, y0, u0, v0, r, g, bcol, depth,
            x0, y1, u0, v1, r, g, bcol, depth,
            x1, y0, u1, v0, r, g, bcol, depth,

            x1, y0, u1, v0, r, g, bcol, depth,
            x0, y1, u0, v1, r, g, bcol, depth,
            x1, y1, u1, v1, r, g, bcol, depth,
        ])



    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_enemies_batch(enemies: list['Enemy']) -> np.ndarray:
    verts: list[float] = []
    for e in enemies:
        if not e.alive:
            continue
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # kvadratisk
        v_shift = int((0.5 - e.height_param) * sprite_h)

        draw_start_y = max(0, -sprite_h // 2 + HALF_H + v_shift)
        draw_end_y   = min(HEIGHT - 1, draw_start_y + sprite_h)
        draw_start_x = -sprite_w // 2 + sprite_screen_x
        draw_end_x   = draw_start_x + sprite_w
        if draw_end_x < 0 or draw_start_x >= WIDTH:
            continue

        draw_start_x = max(0, draw_start_x)
        draw_end_x   = min(WIDTH - 1, draw_end_x)

        x0 = (2.0 * draw_start_x) / WIDTH - 1.0
        x1 = (2.0 * (draw_end_x + 1)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (draw_start_y / HEIGHT)
        y1 = 1.0 - 2.0 * (draw_end_y   / HEIGHT)

        depth = clamp01(trans_y / FAR_PLANE)
        r = g = b = 1.0

        ENEMY_V_FLIP = True  # sett False hvis den blir riktig uten flip
        if ENEMY_V_FLIP:
            u0, v0, u1, v1 = 0.0, 1.0, 1.0, 0.0
        else:
            u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0

        verts.extend([
            x0, y0, u0, v0, r, g, b, depth,
            x0, y1, u0, v1, r, g, b, depth,
            x1, y0, u1, v0, r, g, b, depth,

            x1, y0, u1, v0, r, g, b, depth,
            x0, y1, u0, v1, r, g, b, depth,
            x1, y1, u1, v1, r, g, b, depth,
        ])

    if not verts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_crosshair_quads(size_px: int = 8, thickness_px: int = 2) -> np.ndarray:
    """To små rektangler (horisontalt/vertikalt), sentrert i skjermen."""
    verts: list[float] = []

    def rect_ndc(cx, cy, w, h):
        x0 = (2.0 * (cx - w)) / WIDTH - 1.0
        x1 = (2.0 * (cx + w)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * ((cy - h) / HEIGHT)
        y1 = 1.0 - 2.0 * ((cy + h) / HEIGHT)
        return x0, y0, x1, y1

    r = g = b = 1.0
    depth = 0.0  # helt foran

    # horisontal strek
    x0, y0, x1, y1 = rect_ndc(HALF_W, HALF_H, size_px, thickness_px//2)
    verts.extend([
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,

        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ])

    # vertikal strek
    x0, y0, x1, y1 = rect_ndc(HALF_W, HALF_H, thickness_px//2, size_px)
    verts.extend([
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,

        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ])

    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_weapon_overlay(firing: bool, recoil_t: float) -> np.ndarray:
    """En enkel "pistolboks" nederst (farget quad), m/ liten recoil-bevegelse."""
    base_w, base_h = 200, 120
    x = HALF_W - base_w // 2
    y = HEIGHT - base_h - 10
    if firing:
        y += int(6 * math.sin(min(1.0, recoil_t) * math.pi))

    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + base_w)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + base_h) / HEIGHT)

    # lett gjennomsiktig mørk grå
    # vi bruker v_col for RGB, alpha kommer fra tekstur (1x1 hvit, a=1). For alpha: n.a. her.
    r, g, b = (0.12, 0.12, 0.12)
    depth = 0.0
    verts = [
        x0, y0, 0.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y0, 1.0, 0.0, r, g, b, depth,

        x1, y0, 1.0, 0.0, r, g, b, depth,
        x0, y1, 0.0, 1.0, r, g, b, depth,
        x1, y1, 1.0, 1.0, r, g, b, depth,
    ]
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def build_minimap_quads() -> np.ndarray:
    """Liten GL-basert minimap øverst til venstre.
    - Dører: blå
    - Tilgjengelige vegger: hvit/grå
    - Sett men utilgjengelig: lysegrå prikker
    - Usett utilgjengelig: ikke rendres
    """
    global seen_tiles
    scale = 6
    mm_w = MAP_W * scale
    mm_h = MAP_H * scale
    pad = 10
    verts: list[float] = []

    def add_quad_px(x_px, y_px, w_px, h_px, col, depth):
        r, g, b = col
        x0 = (2.0 * x_px) / WIDTH - 1.0
        x1 = (2.0 * (x_px + w_px)) / WIDTH - 1.0
        y0 = 1.0 - 2.0 * (y_px / HEIGHT)
        y1 = 1.0 - 2.0 * ((y_px + h_px) / HEIGHT)
        verts.extend([
            x0, y0, 0.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x1, y0, 1.0, 0.0, r, g, b, depth,
            x0, y1, 0.0, 1.0, r, g, b, depth,
            x1, y1, 1.0, 1.0, r, g, b, depth,
        ])

    def add_dot_px(cx_px, cy_px, col, depth):
        # liten prikk 2x2 px sentrert i cellen
        r, g, b = col
        x_px = cx_px - 1
        y_px = cy_px - 1
        add_quad_px(x_px, y_px, 2, 2, col, depth)

    # Bakgrunn
    add_quad_px(pad-2, pad-2, mm_w+4, mm_h+4, (0.1, 0.1, 0.1), 0.0)

    # Finn tilgjengelige fliser nå og oppdater 'seen'
    reachable = compute_reachable()
    seen_tiles |= reachable

    def any_neighbor_in(s: set[tuple[int,int]], x: int, y: int) -> bool:
        for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
            if (x+dx, y+dy) in s:
                return True
        return False

    door_blue = (0.2, 0.6, 1.0)
    wall_white = (0.86, 0.86, 0.86)
    seen_dot = (0.7, 0.7, 0.7)

    for y in range(MAP_H):
        for x in range(MAP_W):
            cell = MAP[y][x]
            if cell <= 0:
                continue
            # Skjul vegger i utilgjengelige/uset områder
            accessible = any_neighbor_in(reachable, x, y)
            if not accessible:
                # men om området er sett tidligere (nabo er i seen), tegn prikk
                if any_neighbor_in(seen_tiles, x, y):
                    cx = pad + x*scale + (scale//2)
                    cy = pad + y*scale + (scale//2)
                    add_dot_px(cx, cy, seen_dot, 0.0)
                continue
            # Dører i blått, ellers hvit
            if cell == DOOR_TILE_ID:
                add_quad_px(pad + x*scale, pad + y*scale, scale-1, scale-1, door_blue, 0.0)
            else:
                add_quad_px(pad + x*scale, pad + y*scale, scale-1, scale-1, wall_white, 0.0)

    # Spiller
    px = int(player_x * scale)
    py = int(player_y * scale)
    add_quad_px(pad + px - 2, pad + py - 2, 4, 4, (1.0, 0.3, 0.3), 0.0)

    # Retningsstrek (en liten rektangulær "linje")
    fx = int(px + dir_x * 8)
    fy = int(py + dir_y * 8)
    # tegn som tynn boks mellom (px,py) og (fx,fy)
    # for enkelhet: bare en liten boks på enden
    add_quad_px(pad + fx - 1, pad + fy - 1, 2, 2, (1.0, 0.3, 0.3), 0.0)

    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

# ---------- Input/fysikk ----------
def try_move(nx: float, ny: float) -> tuple[float, float]:
    if not is_wall(int(nx), int(player_y)):
        x = nx
    else:
        x = player_x
    if not is_wall(int(player_x), int(ny)):
        y = ny
    else:
        y = player_y
    return x, y

def handle_input(dt: float) -> None:
    global player_x, player_y, dir_x, dir_y, plane_x, plane_y
    keys = pygame.key.get_pressed()
    rot = 0.0
    if keys[pygame.K_LEFT] or keys[pygame.K_q]:
        rot -= ROT_SPEED * dt
    if keys[pygame.K_RIGHT] or keys[pygame.K_e]:
        rot += ROT_SPEED * dt
    if rot != 0.0:
        cosr, sinr = math.cos(rot), math.sin(rot)
        ndx = dir_x * cosr - dir_y * sinr
        ndy = dir_x * sinr + dir_y * cosr
        npx = plane_x * cosr - plane_y * sinr
        npy = plane_x * sinr + plane_y * cosr
        dir_x, dir_y, plane_x, plane_y = ndx, ndy, npx, npy

    forward = 0.0
    if keys[pygame.K_w] or keys[pygame.K_UP]:
        forward += MOVE_SPEED * dt
    if keys[pygame.K_s] or keys[pygame.K_DOWN]:
        forward -= MOVE_SPEED * dt
    if forward != 0.0:
        nx = player_x + dir_x * forward
        ny = player_y + dir_y * forward
        player_x, player_y = try_move(nx, ny)

    strafe = 0.0
    if keys[pygame.K_a]:
        strafe -= STRAFE_SPEED * dt
    if keys[pygame.K_d]:
        strafe += STRAFE_SPEED * dt
    if strafe != 0.0:
        nx = player_x + (-dir_y) * strafe
        ny = player_y + (dir_x) * strafe
        player_x, player_y = try_move(nx, ny)

def find_door_ahead(max_dist: float = 3.0) -> tuple[int, int] | None:
    # Enkel DDA i sikteretning: finn nærmeste dørcelle i front
    ray_dir_x = dir_x
    ray_dir_y = dir_y
    map_x = int(player_x)
    map_y = int(player_y)
    delta_dist_x = abs(1.0 / ray_dir_x) if ray_dir_x != 0 else 1e30
    delta_dist_y = abs(1.0 / ray_dir_y) if ray_dir_y != 0 else 1e30
    if ray_dir_x < 0:
        step_x = -1
        side_dist_x = (player_x - map_x) * delta_dist_x
    else:
        step_x = 1
        side_dist_x = (map_x + 1.0 - player_x) * delta_dist_x
    if ray_dir_y < 0:
        step_y = -1
        side_dist_y = (player_y - map_y) * delta_dist_y
    else:
        step_y = 1
        side_dist_y = (map_y + 1.0 - player_y) * delta_dist_y
    dist = 0.0
    while dist <= max_dist:
        if side_dist_x < side_dist_y:
            dist = side_dist_x
            side_dist_x += delta_dist_x
            map_x += step_x
        else:
            dist = side_dist_y
            side_dist_y += delta_dist_y
            map_y += step_y
        if not in_map(map_x, map_y):
            return None
        if MAP[map_y][map_x] == DOOR_TILE_ID:
            return (map_x, map_y)
        # Stopp på ordinær vegg, ikke skyt gjennom
        if MAP[map_y][map_x] > 0:
            return None
    return None

def try_open_door() -> None:
    # Finn nærmeste dør i blikkretning og be om åpning
    target = find_door_ahead(max_dist=3.5)
    if target is None:
        return
    d = doors.get(target)
    if d:
        d.request_open()
        print("[Door] Try open", target)

# ---------- Main ----------
def main() -> None:
    pygame.init()
    pygame.display.set_caption("Vibe Wolf (OpenGL)")

    # setup to make it work on mac as well...
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE)

    # Opprett GL-kontekst
    pygame.display.set_mode((WIDTH, HEIGHT), pygame.OPENGL | pygame.DOUBLEBUF)
    gl.glViewport(0, 0, WIDTH, HEIGHT)

    clock = pygame.time.Clock()
    renderer = GLRenderer()

    bullets: list[Bullet] = []
    firing = False
    recoil_t = 0.0

    enemies: list[Enemy] = [
        Enemy(6.5, 10.5),
        Enemy(12.5, 12.5),
        Enemy(16.5, 6.5),
    ]

    # Init dører i kartet: merk MAP-cellen som dør
    MAP[DOOR_Y][DOOR_X] = DOOR_TILE_ID
    doors[(DOOR_X, DOOR_Y)] = Door(DOOR_X, DOOR_Y)
    # Init minimap 'seen' som nåværende tilgjengelig område
    global seen_tiles
    seen_tiles |= compute_reachable()

    # Mus-capture (synlig cursor + crosshair)
    pygame.event.set_grab(True)
    pygame.mouse.set_visible(True)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_TAB:  # eller en annen knapp
                    grab = not pygame.event.get_grab()
                    pygame.event.set_grab(grab)
                    pygame.mouse.set_visible(not grab)
                    print("Mouse grab:", grab)
                if event.key == pygame.K_ESCAPE:
                    running = False
                if event.key == pygame.K_SPACE:
                    bx = player_x + dir_x * 0.4
                    by = player_y + dir_y * 0.4
                    bvx = dir_x * 10.0
                    bvy = dir_y * 10.0
                    bullets.append(Bullet(bx, by, bvx, bvy))
                    firing = True
                    recoil_t = 0.0
                if event.key == pygame.K_z:
                    try_open_door()
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                bx = player_x + dir_x * 0.4
                by = player_y + dir_y * 0.4
                bvx = dir_x * 10.0
                bvy = dir_y * 10.0
                bullets.append(Bullet(bx, by, bvx, bvy))
                firing = True
                recoil_t = 0.0

        handle_input(dt)

        # Oppdater bullets
        for b in bullets:
            b.update(dt)
            if not b.alive:
                continue
            for e in enemies:
                if not e.alive:
                    continue
                dx = e.x - b.x
                dy = e.y - b.y
                if dx * dx + dy * dy <= (e.radius * e.radius):
                    e.alive = False  # fienden "dør"
                    b.alive = False  # kula forbrukes
                    break
        bullets = [b for b in bullets if b.alive]

        # Oppdater fiender
        for e in enemies:
            e.update(dt)

        # Oppdater dør(er)
        for d in doors.values():
            d.update(dt, enemies)

        # ---------- Render ----------
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.05, 0.07, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Bakgrunn (himmel/gulv)
        bg = build_fullscreen_background()
        renderer.draw_arrays(bg, renderer.white_tex, use_tex=False, alpha=False)

        # Vegger (batch pr. tex_id)
        batches_lists = cast_and_build_wall_batches()
        for tid, verts_list in batches_lists.items():
            if not verts_list:
                continue
            arr = np.asarray(verts_list, dtype=np.float32).reshape((-1, 8))
            if tid == 0:
                # uteksturert (bruk white, men uten texture sampling)
                renderer.draw_arrays(arr, renderer.white_tex, use_tex=False, alpha=False)
            elif tid in renderer.textures:
                renderer.draw_arrays(arr, renderer.textures[tid], use_tex=True, alpha=False)

        # Sprites (kuler)
        spr = build_sprites_batch(bullets)
        if spr.size:
            renderer.draw_arrays(spr, renderer.textures[99], use_tex=True, alpha=True)

        # Enemies (billboards)
        enemies_batch = build_enemies_batch(enemies)
        if enemies_batch.size:
            renderer.draw_arrays(enemies_batch, renderer.textures[200], use_tex=True, alpha=True)

        # Crosshair
        cross = build_crosshair_quads(8, 2)
        renderer.draw_arrays(cross, renderer.white_tex, use_tex=False, alpha=False)

        # Weapon overlay
        if firing:
            recoil_t += dt
            if recoil_t > 0.15:
                firing = False
        overlay = build_weapon_overlay(firing, recoil_t)
        renderer.draw_arrays(overlay, renderer.white_tex, use_tex=False, alpha=False)

        # Minimap
        mm = build_minimap_quads()
        renderer.draw_arrays(mm, renderer.white_tex, use_tex=False, alpha=False)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
