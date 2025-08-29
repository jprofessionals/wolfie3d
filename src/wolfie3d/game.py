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
import random
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

# Depth mapping (lineær til [0..1] for gl_FragDepth)
FAR_PLANE = 100.0

# Kart (0=tomt, >0=veggtype/tekstur-id)
MAP: list[list[int]] = [
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,2,0,0,0,0,0,0,0,0,3,0,0,0,0,0,4,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
    [1,0,2,2,2,2,2,0,0,0,0,3,3,3,0,0,4,4,4,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,0,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,0,0,0,0,0,2,0,0,0,0,3,0,0,0,0,4,0,0,1],
    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]
MAP_W = len(MAP[0])
MAP_H = len(MAP)

# Startpos og retning
player_x = 3.5
player_y = 10.5
dir_x, dir_y = 1.0, 0.0
plane_x, plane_y = 0.0, PLANE_LEN

# ---------- Hjelpere ----------
def in_map(ix: int, iy: int) -> bool:
    return 0 <= ix < MAP_W and 0 <= iy < MAP_H

def is_wall(ix: int, iy: int) -> bool:
    return in_map(ix, iy) and MAP[iy][ix] > 0

def clamp01(x: float) -> float:
    if x < 0.0: return 0.0
    if x > 1.0: return 1.0
    return x

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
        self.radius = 0.35
        self.speed = 1.4
        self.height_param = 0.5
        # --- new ---
        self.dying = False           # when True, show happy sprite, then vanish
        self.happy_timer = 0.0       # seconds remaining of happy pose

    def _try_move(self, nx: float, ny: float) -> None:
        if not is_wall(int(nx), int(self.y)):
            self.x = nx
        if not is_wall(int(self.x), int(ny)):
            self.y = ny

    def update(self, dt: float) -> None:
        if not self.alive:
            return
        # If in happy state, count down and don't move
        if self.dying:
            self.happy_timer -= dt
            if self.happy_timer <= 0.0:
                self.alive = False
            return

        # normal chase
        dx = player_x - self.x
        dy = player_y - self.y
        dist = math.hypot(dx, dy) + 1e-9
        if dist > 0.75:
            ux, uy = dx / dist, dy / dist
            step = self.speed * dt
            self._try_move(self.x + ux * step, self.y + uy * step)


def make_cv_stack_surface() -> pygame.Surface:
    """Small stack of CV papers for pickups (RGBA)."""
    W, H = 256, 256
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))

    # soft shadow
    sh = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.ellipse(sh, (0, 0, 0, 90), (60, 190, 136, 34))
    s.blit(sh, (0, 0))

    base = pygame.Rect(72, 72, 112, 140)
    paper = (250, 250, 250, 255)
    edge = (210, 210, 210, 255)

    # 3-4 offset sheets
    for i, off in enumerate([(10, -10), (6, -6), (3, -3), (0, 0)]):
        r = base.move(off)
        a = 255 - i*18
        pygame.draw.rect(s, (paper[0], paper[1], paper[2], a), r, border_radius=6)
        pygame.draw.rect(s, (edge[0], edge[1], edge[2], a), r, width=2, border_radius=6)

    # top "CV" text
    try:
        font = pygame.font.SysFont("Arial", 40, bold=True)
    except Exception:
        font = pygame.font.Font(None, 40)
    txt = font.render("CV", True, (40, 40, 40))
    s.blit(txt, (base.x + 40, base.y + 14))

    return s


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
    """White 'CV' paper sprite with alpha (square texture)."""
    W, H = 128, 128  # bump to 128 for crisper text
    s = pygame.Surface((W, H), pygame.SRCALPHA)
    s.fill((0, 0, 0, 0))

    # soft drop shadow behind paper
    shadow = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow, (0, 0, 0, 80), (20, 78, 88, 36))
    s.blit(shadow, (0, 0))

    # paper sheet
    paper_rect = pygame.Rect(24, 16, 80, 100)
    paper_col = (250, 250, 250, 255)
    edge_col = (210, 210, 210, 255)
    pygame.draw.rect(s, paper_col, paper_rect, border_radius=6)
    pygame.draw.rect(s, edge_col, paper_rect, width=2, border_radius=6)

    # folded corner
    fold = [ (paper_rect.right-18, paper_rect.top+2),
             (paper_rect.right-2,   paper_rect.top+18),
             (paper_rect.right-2,   paper_rect.top+2) ]
    pygame.draw.polygon(s, (235, 235, 235, 255), fold)

    # 'CV' title
    try:
        font = pygame.font.SysFont("Arial", 36, bold=True)
    except Exception:
        font = pygame.font.Font(None, 36)
    text = font.render("CV", True, (40, 40, 40))
    s.blit(text, (paper_rect.x + 24, paper_rect.y + 10))

    # a few lines to suggest content
    x0 = paper_rect.x + 10
    x1 = paper_rect.right - 10
    y = paper_rect.y + 40
    for i in range(5):
        pygame.draw.line(s, (140, 140, 140, 220), (x0, y + i*14), (x1, y + i*14), 2)

    return s

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

def surface_to_texture(surf: pygame.Surface) -> int:
    """Upload a pygame.Surface (with per-pixel alpha) to GL_TEXTURE_2D."""
    # Ensure we have 32-bit RGBA pixels in the Surface *without* display-dependent conversion
    # Avoid convert_alpha() quirks; grab raw pixels in a known order.
    data = pygame.image.tostring(surf, "RGBA", False)  # no flip

    w, h = surf.get_width(), surf.get_height()
    tid = gl.glGenTextures(1)

    gl.glBindTexture(gl.GL_TEXTURE_2D, tid)

    # Be explicit about row alignment (safest for all widths)
    gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

    # Use GL_RGBA as both internal and external format for maximum compatibility
    gl.glTexImage2D(
        gl.GL_TEXTURE_2D, 0, gl.GL_RGBA,
        w, h, 0, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, data
    )

    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)

    gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
    return tid

def make_white_texture() -> int:
    surf = pygame.Surface((1, 1), pygame.SRCALPHA)
    surf.fill((255, 255, 255, 255))
    return surface_to_texture(surf)

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

def make_weapon_empty_hands_surface() -> pygame.Surface:
    """Hands only, no CV stack."""
    W, H = 512, 512
    s = pygame.Surface((W, H), pygame.SRCALPHA)

    # background transparent
    s.fill((0, 0, 0, 0))

    skin = (229, 197, 165, 255)
    shadow = (205, 175, 145, 200)
    sleeve = (40, 60, 120, 255)

    # Left hand
    left_hand = pygame.Rect(120, 360, 120, 90)
    pygame.draw.rect(s, skin, left_hand, border_radius=20)
    pygame.draw.rect(s, shadow, left_hand.inflate(-18, -18), width=0, border_radius=16)
    pygame.draw.ellipse(s, skin, (90, 360, 60, 60))  # thumb
    pygame.draw.rect(s, sleeve, (100, 410, 100, 80), border_radius=16)

    # Right hand
    right_hand = pygame.Rect(272, 360, 120, 90)
    pygame.draw.rect(s, skin, right_hand, border_radius=20)
    pygame.draw.rect(s, shadow, right_hand.inflate(-18, -18), width=0, border_radius=16)
    pygame.draw.ellipse(s, skin, (382, 360, 60, 60))  # thumb
    pygame.draw.rect(s, sleeve, (312, 410, 100, 80), border_radius=16)

    return s


def make_weapon_hands_cv_surface() -> pygame.Surface:
    """Procedurally draw two hands holding a stack of 'CV' papers (RGBA)."""
    W, H = 512, 512
    s = pygame.Surface((W, H), pygame.SRCALPHA)

    # Background transparent
    s.fill((0, 0, 0, 0))

    # --- Paper stack ---
    # Base paper rectangle
    base_rect = pygame.Rect(156, 210, 200, 260)
    paper_col = (245, 245, 245, 255)
    edge_col = (210, 210, 210, 255)
    pygame.draw.rect(s, paper_col, base_rect, border_radius=8)
    pygame.draw.rect(s, edge_col, base_rect, width=2, border_radius=8)

    # Slightly offset layers to look like a stack
    for i, offset in enumerate([(6, -8), (10, -16), (14, -24)]):
        r = base_rect.move(offset[0], offset[1])
        a = max(160, 255 - 30 * i)
        pygame.draw.rect(s, (paper_col[0], paper_col[1], paper_col[2], a), r, border_radius=8)
        pygame.draw.rect(s, (edge_col[0], edge_col[1], edge_col[2], a), r, width=2, border_radius=8)

    # "CV" title on the top sheet
    try:
        font = pygame.font.SysFont("Arial", 72, bold=True)
    except Exception:
        font = pygame.font.Font(None, 72)
    cv_text = font.render("CV", True, (30, 30, 30))
    s.blit(cv_text, (base_rect.x + 70, base_rect.y + 30))

    # A few horizontal lines to mimic text
    line_x = base_rect.x + 24
    for i in range(6):
        y = base_rect.y + 110 + i * 30
        pygame.draw.line(s, (120, 120, 120, 200), (line_x, y), (base_rect.right - 24, y), 3)

    # --- Hands ---
    skin = (229, 197, 165, 255)
    shadow = (205, 175, 145, 200)
    sleeve = (40, 60, 120, 255)

    # Left hand (simple rounded rect + thumb)
    left_hand = pygame.Rect(120, 360, 120, 90)
    pygame.draw.rect(s, skin, left_hand, border_radius=20)
    pygame.draw.rect(s, shadow, left_hand.inflate(-18, -18), width=0, border_radius=16)
    # Left thumb
    pygame.draw.ellipse(s, skin, (90, 360, 60, 60))
    # Left sleeve
    pygame.draw.rect(s, sleeve, (100, 410, 100, 80), border_radius=16)

    # Right hand
    right_hand = pygame.Rect(272, 360, 120, 90)
    pygame.draw.rect(s, skin, right_hand, border_radius=20)
    pygame.draw.rect(s, shadow, right_hand.inflate(-18, -18), width=0, border_radius=16)
    # Right thumb
    pygame.draw.ellipse(s, skin, (382, 360, 60, 60))
    # Right sleeve
    pygame.draw.rect(s, sleeve, (312, 410, 100, 80), border_radius=16)

    # Small drop shadow under papers (for depth)
    shadow_surf = pygame.Surface((W, H), pygame.SRCALPHA)
    pygame.draw.ellipse(shadow_surf, (0, 0, 0, 80), (160, 440, 192, 40))
    s.blit(shadow_surf, (0, 0))

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

        # Weapon sprite (procedural)
        self.weapon_tex = surface_to_texture(make_weapon_hands_cv_surface())
        self.weapon_w = 512
        self.weapon_h = 512

        self.weapon_empty_tex = surface_to_texture(make_weapon_empty_hands_surface())


        # Ammo pickup sprite
        self.textures[150] = surface_to_texture(make_cv_stack_surface())


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
        return surface_to_texture(surf)

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
            tex_id = surface_to_texture(surf)
            print(f"[GLRenderer]  - OK (GL tex id {tex_id})")
            return tex_id

        self.textures[1] = _load(files[1], 512)
        self.textures[2] = _load(files[2], 512)
        self.textures[3] = _load(files[3], 512)
        self.textures[4] = _load(files[4], 512)

        # Sprite (kule) – behold prosedyre
        self.textures[99] = surface_to_texture(make_bullet_texture())

        # Enemy sprite (ID 200): prøv fil, ellers prosedyral placeholder
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            enemy_path = sprites_dir / "sjefen.png"
            print(f"[GLRenderer] Leter etter enemy sprite i: {enemy_path}")
            if enemy_path.exists():
                self.textures[200] = self._load_texture_file(enemy_path, 512)
                print(f"[GLRenderer] Enemy OK (GL tex id {self.textures[200]})")
            else:
                # fallback – prosedural fiende
                self.textures[200] = surface_to_texture(make_enemy_texture())
                print("[GLRenderer] Enemy: bruker prosedural sprite")
        except Exception as ex:
            print(f"[GLRenderer] Enemy: FEIL ved lasting ({ex}), bruker prosedural")
            self.textures[200] = surface_to_texture(make_enemy_texture())

        # Enemy HAPPY sprite (ID 201): assets/sprites/sjefenglad.png
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            happy_path = sprites_dir / "sjefenglad.png"
            print(f"[GLRenderer] Leter etter happy sprite i: {happy_path}")
            if happy_path.exists():
                self.textures[201] = self._load_texture_file(happy_path, 512)
                print(f"[GLRenderer] Happy enemy OK (GL tex id {self.textures[201]})")
            else:
                # fallback: reuse normal enemy if not present
                self.textures[201] = self.textures.get(200, surface_to_texture(make_enemy_texture()))
                print("[GLRenderer] Happy enemy: fant ikke fil, bruker fallback")
        except Exception as ex:
            print(f"[GLRenderer] Happy enemy: FEIL ved lasting ({ex}), bruker fallback")
            self.textures[201] = self.textures.get(200, surface_to_texture(make_enemy_texture()))


        # Kill screen (full-screen image)
        try:
            sprites_dir = self._resolve_textures_base().parent / "sprites"
            end_path = sprites_dir / "end.png"
            print(f"[GLRenderer] Leter etter end screen i: {end_path}")
            if end_path.exists():
                self.textures[999] = self._load_texture_file(end_path, size=1024)
                print(f"[GLRenderer] End screen OK (GL tex id {self.textures[999]})")
            else:
                print("[GLRenderer] End screen not found; fallback to white")
                self.textures[999] = self.white_tex
        except Exception as ex:
            print(f"[GLRenderer] End screen: FEIL ved lasting ({ex})")
            self.textures[999] = self.white_tex


        print("[GLRenderer] Teksturer lastet.\n")

    # ---------- draw ----------
    def draw_arrays(self, verts: np.ndarray, texture: int, use_tex: bool) -> None:
        if verts.size == 0:
            return
        gl.glUseProgram(self.prog)
        gl.glUniform1i(self.uni_use_tex, 1 if use_tex else 0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, texture if use_tex else self.white_tex)

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
    return x_left, x_right

def y_ndc(y_pix: int) -> float:
    """Konverter skjerm-Y (0 top) til NDC-Y (1 top, -1 bunn)."""
    return 1.0 - 2.0 * (y_pix / float(HEIGHT))

def dim_for_side(side: int) -> float:
    # dim litt på sidevegger (liknende BLEND_MULT tidligere)
    return 0.78 if side == 1 else 1.0

def build_pickups_batch(pickups: list['Pickup']) -> np.ndarray:
    """Billboard quads for floor pickups (stacks of CVs)."""
    verts: list[float] = []
    for p in pickups:
        if not p.alive:
            continue
        spr_x = p.x - player_x
        spr_y = p.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))

        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h  # square sprite texture
        v_shift = int((0.5 - p.height_param) * sprite_h)  # place near floor

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


def cast_and_build_wall_batches() -> dict[int, list[float]]:
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
            if MAP[map_y][map_x] > 0:
                hit = 1
                tex_id = MAP[map_y][map_x]

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

        # skjermhøyde på vegg
        line_height = int(HEIGHT / (perp_wall_dist + 1e-6))
        draw_start = max(0, -line_height // 2 + HALF_H)
        draw_end = min(HEIGHT - 1, line_height // 2 + HALF_H)

        # NDC koordinater for 1-px bred stripe
        x_left, x_right = column_ndc(x)
        top_ndc = y_ndc(draw_start)
        bot_ndc = y_ndc(draw_end)

        # Farge-dim (samme på hele kolonnen)
        c = dim_for_side(side)
        r = g = b = c

        # Depth som lineær [0..1] (0 = nærmest)
        depth = clamp01(perp_wall_dist / FAR_PLANE)

        # 2 triangler (6 vertikser). Vertex-layout:
        # [x, y, u, v, r, g, b, depth]
        v = [
            # tri 1
            x_left,  top_ndc, u, 0.0, r, g, b, depth,
            x_left,  bot_ndc, u, 1.0, r, g, b, depth,
            x_right, top_ndc, u, 0.0, r, g, b, depth,
            # tri 2
            x_right, top_ndc, u, 0.0, r, g, b, depth,
            x_left,  bot_ndc, u, 1.0, r, g, b, depth,
            x_right, bot_ndc, u, 1.0, r, g, b, depth,
        ]
        batches.setdefault(tex_id, []).extend(v)
    return batches

def build_fullscreen_quad() -> np.ndarray:
    """Fullscreen quad with UVs (0..1)."""
    verts = [
        -1.0,  1.0, 0.0, 0.0, 1,1,1,0,
        -1.0, -1.0, 0.0, 1.0, 1,1,1,0,
         1.0,  1.0, 1.0, 0.0, 1,1,1,0,

         1.0,  1.0, 1.0, 0.0, 1,1,1,0,
        -1.0, -1.0, 0.0, 1.0, 1,1,1,0,
         1.0, -1.0, 1.0, 1.0, 1,1,1,0,
    ]
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


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

def build_enemies_batch_filtered(enemies: list['Enemy'], use_happy: bool) -> np.ndarray:
    verts: list[float] = []
    for e in enemies:
        if not e.alive:
            continue
        # choose which set we’re building
        if use_happy and not e.dying:
            continue
        if (not use_happy) and e.dying:
            continue

        # --- same body as your build_enemies_batch(), unchanged below ---
        spr_x = e.x - player_x
        spr_y = e.y - player_y
        inv_det = 1.0 / (plane_x * dir_y - dir_x * plane_y + 1e-9)
        trans_x = inv_det * (dir_y * spr_x - dir_x * spr_y)
        trans_y = inv_det * (-plane_y * spr_x + plane_x * spr_y)
        if trans_y <= 0:
            continue

        sprite_screen_x = int((WIDTH / 2) * (1 + trans_x / trans_y))
        sprite_h = abs(int(HEIGHT / trans_y))
        sprite_w = sprite_h
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

        ENEMY_V_FLIP = False  # matches your current no-flip upload path
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

        ENEMY_V_FLIP = False  # sett False hvis den blir riktig uten flip
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

def build_weapon_sprite_overlay(tex_w: int, tex_h: int, firing: bool, recoil_t: float) -> np.ndarray:
    """
    Place a textured quad (the 'weapon') bottom-center, with a tiny vertical recoil.
    """
    # Desired on-screen size
    on_w = 320
    on_h = int(on_w * (tex_h / tex_w))  # keep aspect

    x = HALF_W - on_w // 2
    y = HEIGHT - on_h - 12

    if firing:
        y += int(6 * math.sin(min(1.0, recoil_t) * math.pi))

    x0 = (2.0 * x) / WIDTH - 1.0
    x1 = (2.0 * (x + on_w)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y / HEIGHT)
    y1 = 1.0 - 2.0 * ((y + on_h) / HEIGHT)

    r = g = b = 1.0
    depth = 0.0
    u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0

    verts = [
        x0, y0, u0, v0, r, g, b, depth,
        x0, y1, u0, v1, r, g, b, depth,
        x1, y0, u1, v0, r, g, b, depth,

        x1, y0, u1, v0, r, g, b, depth,
        x0, y1, u0, v1, r, g, b, depth,
        x1, y1, u1, v1, r, g, b, depth,
    ]
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))


def build_minimap_quads() -> np.ndarray:
    """Liten GL-basert minimap øverst til venstre."""
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

    # Bakgrunn
    add_quad_px(pad-2, pad-2, mm_w+4, mm_h+4, (0.1, 0.1, 0.1), 0.0)

    # Celler
    for y in range(MAP_H):
        for x in range(MAP_W):
            if MAP[y][x] > 0:
                col = (0.86, 0.86, 0.86)
                add_quad_px(pad + x*scale, pad + y*scale, scale-1, scale-1, col, 0.0)

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

class Pickup:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.alive = True
        # render height (where the billboard centers vertically)
        self.height_param = 0.35

def random_empty_cell(min_dist_from_player: float = 3.0) -> tuple[float, float] | None:
    """Pick a random empty map cell not too close to the player."""
    empties = [(x, y) for y in range(MAP_H) for x in range(MAP_W) if MAP[y][x] == 0]
    random.shuffle(empties)
    for x, y in empties:
        cx, cy = x + 0.5, y + 0.5
        if math.hypot(cx - player_x, cy - player_y) >= min_dist_from_player:
            return cx, cy
    return None

def make_text_surface(text: str, font_size: int = 28) -> pygame.Surface:
    """Render text onto an RGBA surface (tight box)."""
    try:
        font = pygame.font.SysFont("DejaVu Sans, Arial", font_size, bold=True)
    except Exception:
        font = pygame.font.Font(None, font_size)
    s = font.render(text, True, (255, 255, 255))
    # add slight shadow for readability
    w, h = s.get_width(), s.get_height()
    out = pygame.Surface((w+2, h+2), pygame.SRCALPHA)
    shadow = font.render(text, True, (0, 0, 0))
    out.blit(shadow, (2, 2))
    out.blit(s, (0, 0))
    return out

def build_hud_quad_px(x_px: int, y_px: int, w_px: int, h_px: int) -> np.ndarray:
    """Screen-space textured quad at (x,y) top-left in pixels."""
    x0 = (2.0 * x_px) / WIDTH - 1.0
    x1 = (2.0 * (x_px + w_px)) / WIDTH - 1.0
    y0 = 1.0 - 2.0 * (y_px / HEIGHT)
    y1 = 1.0 - 2.0 * ((y_px + h_px) / HEIGHT)
    r = g = b = 1.0
    depth = 0.0
    u0, v0, u1, v1 = 0.0, 0.0, 1.0, 1.0
    verts = [
        x0, y0, u0, v0, r, g, b, depth,
        x0, y1, u0, v1, r, g, b, depth,
        x1, y0, u1, v0, r, g, b, depth,
        x1, y0, u1, v0, r, g, b, depth,
        x0, y1, u0, v1, r, g, b, depth,
        x1, y1, u1, v1, r, g, b, depth,
    ]
    return np.asarray(verts, dtype=np.float32).reshape((-1, 8))

def _first_empty_from_corner(corner: str) -> tuple[float, float] | None:
    """
    Find the first empty MAP cell scanning inwards from a corner,
    and return its center (x+0.5, y+0.5).
    Corners: 'tl','tr','bl','br'
    """
    xs = range(1, MAP_W-1)
    ys = range(1, MAP_H-1)
    if corner == 'tl':
        xr, yr = xs, ys
    elif corner == 'tr':
        xr, yr = reversed(xs), ys
    elif corner == 'bl':
        xr, yr = xs, reversed(ys)
    elif corner == 'br':
        xr, yr = reversed(xs), reversed(ys)
    else:
        return None

    for y in yr:
        for x in xr:
            if MAP[y][x] == 0:
                return (x + 0.5, y + 0.5)
    return None

def corner_spawn_positions() -> list[tuple[float, float]]:
    """Return up to four spawn positions, one per corner (skips if no empty cell found)."""
    pos = []
    for c in ('tl', 'tr', 'bl', 'br'):
        p = _first_empty_from_corner(c)
        if p:
            pos.append(p)
    return pos

def far_from_player(x: float, y: float, min_dist: float = 3.0) -> bool:
    return math.hypot(x - player_x, y - player_y) >= min_dist



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

    ammo = 0
    print(f"[AMMO] {ammo}")

    # --- HUD (ammo text) ---
    ammo_tex = None
    ammo_w = ammo_h = 0
    last_ammo = None

    def refresh_ammo_tex():
        nonlocal ammo_tex, ammo_w, ammo_h, last_ammo
        if last_ammo == ammo:
            return
        surf = make_text_surface(f"AMMO: {ammo}", 28)
        ammo_w, ammo_h = surf.get_width(), surf.get_height()
        # delete old GL texture if any
        if ammo_tex:
            gl.glDeleteTextures(int(ammo_tex))
        ammo_tex = surface_to_texture(surf)
        last_ammo = ammo

    refresh_ammo_tex()

    pickups: list[Pickup] = []
    spawn_timer = 0.0
    SPAWN_INTERVAL = 10.0  # seconds
    PICKUP_RADIUS = 0.6    # distance to collect


    bullets: list[Bullet] = []
    firing = False
    recoil_t = 0.0

    enemies: list[Enemy] = []
    enemy_spawn_timer = 0.0
    ENEMY_SPAWN_INTERVAL = 5.0   # spawn more often (tweak as you like)
    ENEMY_MAX = 12               # cap population
    ENEMY_MIN_PLAYER_DIST = 3.0  # don't spawn too close

    for (sx, sy) in corner_spawn_positions():
        if far_from_player(sx, sy, ENEMY_MIN_PLAYER_DIST):
            enemies.append(Enemy(sx, sy))


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
                    if ammo > 0:
                        bx = player_x + dir_x * 0.4
                        by = player_y + dir_y * 0.4
                        bvx = dir_x * 10.0
                        bvy = dir_y * 10.0
                        bullets.append(Bullet(bx, by, bvx, bvy))
                        ammo -= 1
                        print(f"[AMMO] {ammo}")
                        refresh_ammo_tex()
                        firing = True
                        recoil_t = 0.0
                    else:
                        print("[AMMO] empty")

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if ammo > 0:
                    bx = player_x + dir_x * 0.4
                    by = player_y + dir_y * 0.4
                    bvx = dir_x * 10.0
                    bvy = dir_y * 10.0
                    bullets.append(Bullet(bx, by, bvx, bvy))
                    ammo -= 1
                    print(f"[AMMO] {ammo}")
                    firing = True
                    recoil_t = 0.0
                else:
                    print("[AMMO] empty")

        handle_input(dt)

        # Timed enemy spawns from random corner
        enemy_spawn_timer += dt
        if enemy_spawn_timer >= ENEMY_SPAWN_INTERVAL:
            enemy_spawn_timer -= ENEMY_SPAWN_INTERVAL
            if sum(1 for e in enemies if e.alive or e.dying) < ENEMY_MAX:
                corners = corner_spawn_positions()
                random.shuffle(corners)
                for (sx, sy) in corners:
                    # avoid spawning on top of another enemy (simple radius check)
                    too_close = any((math.hypot(e.x - sx, e.y - sy) < 1.0) for e in enemies if e.alive or e.dying)
                    if far_from_player(sx, sy, ENEMY_MIN_PLAYER_DIST) and not too_close:
                        enemies.append(Enemy(sx, sy))
                        break  # spawn one this tick


        # Spawn timer
        spawn_timer += dt
        if spawn_timer >= SPAWN_INTERVAL:
            spawn_timer -= SPAWN_INTERVAL
            pos = random_empty_cell(min_dist_from_player=3.0)
            if pos is not None:
                px, py = pos
                pickups.append(Pickup(px, py))
                # (optional) print where it spawned
                # print(f"[PICKUP] spawned at {px:.1f}, {py:.1f}")


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
                    if not e.dying:              # only trigger once
                        e.dying = True
                        e.happy_timer = 2.0      # show sjefenglad for 2 seconds
                    b.alive = False              # consume projectile
                    break
        bullets = [b for b in bullets if b.alive]

        # Oppdater fiender
        for e in enemies:
            e.update(dt)

        # Check if any enemy touches the player
        for e in enemies:
            if e.alive and not e.dying:
                if math.hypot(e.x - player_x, e.y - player_y) <= e.radius + 0.2:
                    # Trigger kill screen
                    gl.glClearColor(0,0,0,1)
                    gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                    quad = build_fullscreen_quad()
                    renderer.draw_arrays(quad, renderer.textures[999], use_tex=True)
                    pygame.display.flip()
                    pygame.time.wait(20000)   # show for 2 sec (adjust as you like)
                    running = False
                    break


        # Collect if close
        for p in pickups:
            if not p.alive:
                continue
            if math.hypot(p.x - player_x, p.y - player_y) <= PICKUP_RADIUS:
                p.alive = False
                ammo += 3
                refresh_ammo_tex()
                print(f"[AMMO] +3 -> {ammo}")
        # Compact list
        pickups = [p for p in pickups if p.alive]


        # ---------- Render ----------
        gl.glViewport(0, 0, WIDTH, HEIGHT)
        gl.glClearColor(0.05, 0.07, 0.1, 1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Bakgrunn (himmel/gulv)
        bg = build_fullscreen_background()
        renderer.draw_arrays(bg, renderer.white_tex, use_tex=False)

        # Vegger (batch pr. tex_id)
        batches_lists = cast_and_build_wall_batches()
        for tid, verts_list in batches_lists.items():
            if tid not in renderer.textures:
                continue
            if not verts_list:
                continue
            arr = np.asarray(verts_list, dtype=np.float32).reshape((-1, 8))
            renderer.draw_arrays(arr, renderer.textures[tid], use_tex=True)

        # Sprites (kuler)
        spr = build_sprites_batch(bullets)
        if spr.size:
            renderer.draw_arrays(spr, renderer.textures[99], use_tex=True)

        # Enemies (billboards)
        # Enemies: normal + happy (two batches, two textures)
        normal_batch = build_enemies_batch_filtered(enemies, use_happy=False)
        if normal_batch.size:
            renderer.draw_arrays(normal_batch, renderer.textures[200], use_tex=True)

        happy_batch = build_enemies_batch_filtered(enemies, use_happy=True)
        if happy_batch.size:
            renderer.draw_arrays(happy_batch, renderer.textures[201], use_tex=True)

        # Ammo pickups
        pk = build_pickups_batch(pickups)
        if pk.size:
            renderer.draw_arrays(pk, renderer.textures[150], use_tex=True)


        # Crosshair
        cross = build_crosshair_quads(8, 2)
        renderer.draw_arrays(cross, renderer.white_tex, use_tex=False)

        # Weapon overlay
        # Weapon sprite overlay (hands + CV papers)
        if firing:
            recoil_t += dt
            if recoil_t > 0.15:
                firing = False

        weapon_verts = build_weapon_sprite_overlay(renderer.weapon_w, renderer.weapon_h, firing, recoil_t)
        if ammo > 0:
            tex = renderer.weapon_tex
        else:
            tex = renderer.weapon_empty_tex
        renderer.draw_arrays(weapon_verts, tex, use_tex=True)

        # --- HUD: draw ammo text top-left (padding 10 px) ---
        if ammo_tex:
            gl.glDepthMask(gl.GL_FALSE)        # don't write depth for UI
            hud = build_hud_quad_px(10, 10, ammo_w, ammo_h)
            renderer.draw_arrays(hud, ammo_tex, use_tex=True)
            gl.glDepthMask(gl.GL_TRUE)

        # Minimap
        mm = build_minimap_quads()
        renderer.draw_arrays(mm, renderer.white_tex, use_tex=False)

        pygame.display.flip()

    pygame.quit()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("Fatal error:", e)
        sys.exit(1)
