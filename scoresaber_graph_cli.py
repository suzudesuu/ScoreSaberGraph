import sys
import json
import os
import io
from typing import List

import requests
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


API_PLAYER_FULL = "https://scoresaber.com/api/player/{player_id}/full"


def fetch_player_full(player_id: str) -> dict:
    url = API_PLAYER_FULL.format(player_id=player_id)
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    # В некоторых версиях API всё лежит в playerInfo
    return data.get("playerInfo", data)


def parse_histories(player_info: dict) -> List[int]:
    """
    В ScoreSaber в поле histories лежит строка вида '123,120,118,...'
    Это история глобального ранга за последние дни.
    """
    hist = player_info.get("histories") or ""
    hist = hist.strip()
    if not hist:
        return []

    parts = [p for p in hist.split(",") if p.strip() != ""]
    ranks = []
    for p in parts:
        try:
            ranks.append(int(p))
        except ValueError:
            pass
    return ranks

def catmull_rom_spline(x, y, samples=10):
    """
    Catmull-Rom spline interpolation (без SciPy)
    """
    x_new = []
    y_new = []
    for i in range(len(x) - 1):
        p0x = x[i - 1] if i - 1 >= 0 else x[i]
        p1x = x[i]
        p2x = x[i + 1]
        p3x = x[i + 2] if i + 2 < len(x) else x[i + 1]

        p0y = y[i - 1] if i - 1 >= 0 else y[i]
        p1y = y[i]
        p2y = y[i + 1]
        p3y = y[i + 2] if i + 2 < len(y) else y[i + 1]

        for t in np.linspace(0, 1, samples):
            t2 = t * t
            t3 = t2 * t
            fx = 0.5 * (
                (2 * p1x)
                + (-p0x + p2x) * t
                + (2 * p0x - 5 * p1x + 4 * p2x - p3x) * t2
                + (-p0x + 3 * p1x - 3 * p2x + p3x) * t3
            )
            fy = 0.5 * (
                (2 * p1y)
                + (-p0y + p2y) * t
                + (2 * p0y - 5 * p1y + 4 * p2y - p3y) * t2
                + (-p0y + 3 * p1y - 3 * p2y + p3y) * t3
            )
            x_new.append(fx)
            y_new.append(fy)

    return np.array(x_new), np.array(y_new)

def make_rank_graph_image(
    ranks: List[int],
    background: str = "dark",
    width_px: int = 1600,
    height_px: int = 450,
) -> Image.Image:
    """
    График ранга:
    - плавная линия (Catmull-Rom)
    - мягкое свечение вокруг линии через blur в Pillow
    """
    if not ranks:
        ranks = [200] * 10

    # исходные точки
    x = np.arange(len(ranks))
    y = np.array(ranks, dtype=float)

    # плавная кривая Catmull-Rom
    if len(ranks) > 1:
        x_smooth, y_smooth = catmull_rom_spline(x, y, samples=20)
    else:
        x_smooth, y_smooth = x, y

    dpi = 100
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    bg_color = "#0f172a"
    if background == "transparent":
        ax.set_facecolor("none")
        fig.patch.set_alpha(0.0)
    else:
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)

    # тёмная сетка
    grid_color = "#00000055"
    ax.yaxis.grid(True, color=grid_color, linewidth=0.7)
    ax.xaxis.grid(True, color=grid_color, linewidth=0.7)

    # одна базовая линия (без псевдо‑glow)
    line_color = "#4CC9F0"
    ax.plot(
        x_smooth,
        y_smooth,
        color=line_color,
        linewidth=3.0,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    ax.invert_yaxis()

    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", colors="#A8B3CF", labelsize=8, length=0)
    ax.tick_params(axis="y", colors="#A8B3CF", labelsize=8, length=0)
    ax.set_xticklabels([])

    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.08)

    # ---- рендерим в буфер ----
    buf = io.BytesIO()
    if background == "transparent":
        fig.savefig(buf, format="png", transparent=True)
    else:
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGBA")

    # ---- НАСТОЯЩЕЕ GLOW ЧЕРЕЗ BLUR ----
    # Берём альфа‑канал (где нарисованы линия + сетка)
    r, g, b, a = img.split()

    # цвет свечения
    glow_color = (76, 201, 240, 255)  # тот же #4CC9F0

    # заполняем этим цветом область, где есть альфа
    glow_base = Image.new("RGBA", img.size, glow_color)
    glow = Image.new("RGBA", img.size, (0, 0, 0, 0))
    glow.paste(glow_base, mask=a)

    # размытие — ширина и мягкость свечения
    glow = glow.filter(ImageFilter.GaussianBlur(radius=12))

    # можно чуть ослабить свечение, если будет слишком ярко:
    # glow.putalpha(200)

    # кладём glow под исходный график
    result = Image.alpha_composite(glow, img)

    return result

def try_load_font(size: int) -> ImageFont.FreeTypeFont:
    # Пытаемся найти BOLD‑шрифт
    possible = [
        "C:/Windows/Fonts/segoeuib.ttf",          # Segoe UI Bold
        "C:/Windows/Fonts/arialbd.ttf",           # Arial Bold
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ]
    for path in possible:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()

def draw_bold_text(draw, xy, text, font, fill, stroke_fill=None, stroke_width=1):
    """
    Примитивный "жирный" текст: рисуем несколько раз вокруг, потом основной.
    Работает даже если нет отдельного bold-шрифта.
    """
    if stroke_fill is None:
        stroke_fill = fill

    x, y = xy
    offsets = [(-stroke_width, 0), (stroke_width, 0),
               (0, -stroke_width), (0, stroke_width)]
    for dx, dy in offsets:
        draw.text((x + dx, y + dy), text, font=font, fill=stroke_fill)

    draw.text((x, y), text, font=font, fill=fill)

def make_player_header_image(player_info: dict, width_px: int) -> Image.Image:
    """
    Верхняя карточка игрока: крупная аватарка + жирный крупный текст.
    Размеры заметно увеличены относительно предыдущей версии.
    """
    height_px = 260  # ещё выше
    card_radius = 34

    card = Image.new("RGBA", (width_px, height_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(card)

    avatar_url = (
        player_info.get("profilePicture")
        or player_info.get("avatar")
        or player_info.get("profile")
        or ""
    )

    ava_img = None
    if avatar_url:
        try:
            r = requests.get(avatar_url, timeout=10)
            r.raise_for_status()
            ava_img = Image.open(io.BytesIO(r.content)).convert("RGBA")
        except Exception:
            ava_img = None

    # фон из аватарки, без искажения пропорций
    if ava_img is not None:
        aw, ah = ava_img.size
        scale = max(width_px / aw, height_px / ah) * 1.4
        new_w = int(aw * scale)
        new_h = int(ah * scale)
        bg = ava_img.resize((new_w, new_h), Image.LANCZOS)

        left = (new_w - width_px) // 2
        top = (new_h - height_px) // 2
        bg = bg.crop((left, top, left + width_px, top + height_px))

        bg = bg.filter(ImageFilter.GaussianBlur(radius=10))
        dark_layer = Image.new("RGBA", (width_px, height_px), (10, 10, 25, 170))
        bg = Image.alpha_composite(bg, dark_layer)

        mask = Image.new("L", (width_px, height_px), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle(
            [0, 0, width_px - 1, height_px - 1],
            radius=card_radius,
            fill=255,
        )

        card = Image.composite(bg, card, mask)
        draw = ImageDraw.Draw(card)
    else:
        draw.rounded_rectangle(
            [0, 0, width_px - 1, height_px - 1],
            radius=card_radius,
            fill=(15, 23, 42, 235),
        )

    # лёгкая вуаль поверх
    overlay = Image.new("RGBA", (width_px, height_px), (10, 10, 25, 110))
    card = Image.alpha_composite(card, overlay)
    draw = ImageDraw.Draw(card)

    # --- аватар слева (ещё больше) ---
    ava_box_size = 180  # было 128
    ava_padding = 38
    if ava_img is not None:
        ava_small = ava_img.resize((ava_box_size, ava_box_size), Image.LANCZOS)

        mask = Image.new("L", (ava_box_size, ava_box_size), 0)
        mdraw = ImageDraw.Draw(mask)
        mdraw.rounded_rectangle(
            [0, 0, ava_box_size - 1, ava_box_size - 1],
            radius=28,
            fill=255,
        )

        card.paste(ava_small, (ava_padding, ava_padding), mask)

    # --- текст (жирный и крупный) ---
    name_font = try_load_font(64)
    small_font = try_load_font(32)
    right_big_font = try_load_font(60)
    right_small_font = try_load_font(32)

    name = player_info.get("name") or "Unknown player"
    country = player_info.get("country") or "??"
    c_rank = player_info.get("countryRank") or "#?"
    rank = player_info.get("rank") or player_info.get("globalRank") or "?"
    pp = player_info.get("pp") or player_info.get("playerPp") or 0

    text_x = ava_padding + ava_box_size + 28
    text_y = ava_padding + 20

    # имя
    draw_bold_text(
        draw,
        (text_x, text_y),
        name,
        font=name_font,
        fill=(248, 250, 252, 255),
        stroke_fill=(15, 23, 42, 200),
        stroke_width=1,
    )

    # страна + локальный ранг
    meta_str = f"{country} — #{c_rank}"
    draw_bold_text(
        draw,
        (text_x, text_y + 72),
        meta_str,
        font=small_font,
        fill=(226, 232, 240, 230),
        stroke_fill=(15, 23, 42, 180),
        stroke_width=1,
    )

    # правый блок: глобальный ранг + pp

    text_rank = f"#{rank}"
    text_pp   = f"{round(pp)} pp"

    # измеряем ширину текста через draw.textbbox,
    # чтобы не упираться в правый край
    rank_bbox = draw.textbbox((0, 0), text_rank, font=right_big_font)
    pp_bbox   = draw.textbbox((0, 0), text_pp,   font=right_small_font)

    rank_w = rank_bbox[2] - rank_bbox[0]
    pp_w   = pp_bbox[2] - pp_bbox[0]

    block_w = max(rank_w, pp_w)

    # твой базовый отступ
    base_x = width_px - 190

    # минимальный отступ от правого края
    safe_margin = 48
    max_right = width_px - safe_margin

    if base_x + block_w <= max_right:
        right_x = base_x
    else:
        right_x = max_right - block_w

    draw_bold_text(
        draw,
        (right_x, ava_padding + 30),
        text_rank,
        font=right_big_font,
        fill=(248, 250, 252, 255),
        stroke_fill=(15, 23, 42, 200),
        stroke_width=1,
    )

    draw_bold_text(
        draw,
        (right_x, ava_padding + 96),
        text_pp,
        font=right_small_font,
        fill=(226, 232, 240, 230),
        stroke_fill=(15, 23, 42, 180),
        stroke_width=1,
    )

    return card

def make_graph_avatar_bg(player_info: dict, width: int, height: int) -> Image.Image:
    """
    Фон для области графика из аватарки игрока:
    растягиваем без искажения пропорций, блюрим и затемняем.
    """
    avatar_url = (
        player_info.get("profilePicture")
        or player_info.get("avatar")
        or player_info.get("profile")
        or ""
    )

    bg = Image.new("RGBA", (width, height), (10, 10, 25, 255))

    if not avatar_url:
        # если нет аватарки — тупо тёмный фон
        return bg

    try:
        r = requests.get(avatar_url, timeout=10)
        r.raise_for_status()
        ava_img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    except Exception:
        return bg

    aw, ah = ava_img.size
    # скейлим, чтобы картинка полностью закрывала область графика
    scale = max(width / aw, height / ah) * 1.4
    new_w = int(aw * scale)
    new_h = int(ah * scale)
    big = ava_img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - width) // 2
    top = (new_h - height) // 2
    big = big.crop((left, top, left + width, top + height))

    # сильный blur + затемнение
    big = big.filter(ImageFilter.GaussianBlur(radius=12))
    dark_layer = Image.new("RGBA", (width, height), (5, 10, 25, 210))
    big = Image.alpha_composite(big, dark_layer)

    return big

def compose_both_image(
    player_info: dict,
    ranks: List[int],
    output_path: str,
) -> str:
    """
    Итоговая картинка: сверху карточка, снизу график.
    Фон внешнего полотна прозрачный, есть отступы.
    Нижний блок: фон из аватарки (blur + dark), сверху график.
    """
    inner_width = 1600
    graph_height = 450

    # верхняя карточка игрока
    header_img = make_player_header_image(player_info, inner_width)
    header_w, header_h = header_img.size

    # сам график (с glow), фон у него прозрачный
    graph_img = make_rank_graph_image(
        ranks,
        background="transparent",
        width_px=inner_width,
        height_px=graph_height,
    )

    # отступы вокруг
    margin_x = 48
    margin_y_top = 36
    margin_y_bottom = 48
    between = 26  # расстояние между шапкой и графиком

    total_width = header_w + margin_x * 2
    total_height = header_h + graph_height + between + margin_y_top + margin_y_bottom

    canvas = Image.new("RGBA", (total_width, total_height), (0, 0, 0, 0))

    header_x = margin_x
    header_y = margin_y_top
    graph_x = margin_x
    graph_y = margin_y_top + header_h + between

    radius = 30

    def rounded_mask(w, h, r):
        m = Image.new("L", (w, h), 0)
        d = ImageDraw.Draw(m)
        d.rounded_rectangle([0, 0, w - 1, h - 1], radius=r, fill=255)
        return m

    # --- верхняя карточка ---
    header_mask = rounded_mask(header_w, header_h, radius)
    header_bg = Image.new("RGBA", (header_w, header_h), (15, 23, 42, 0))
    header_bg = Image.alpha_composite(header_bg, header_img)
    canvas.paste(header_bg, (header_x, header_y), header_mask)

    # --- нижний блок: фон из аватарки + график сверху ---
    avatar_bg = make_graph_avatar_bg(player_info, inner_width, graph_height)
    graph_mask = rounded_mask(inner_width, graph_height, radius)
    graph_bg = Image.alpha_composite(avatar_bg, graph_img)
    canvas.paste(graph_bg, (graph_x, graph_y), graph_mask)

    canvas.save(output_path, format="PNG")
    return output_path

def make_rank_graph_image(
    ranks: List[int],
    background: str = "dark",
    width_px: int = 1600,
    height_px: int = 450,
) -> Image.Image:
    """
    График ранга в стиле ScoreSaber:
    - плавная линия (Catmull-Rom spline)
    - многослойный glow вокруг линии
    """
    if not ranks:
        ranks = [200] * 10

    # исходные точки
    x = np.arange(len(ranks))
    y = np.array(ranks, dtype=float)

    # плавная кривая Catmull-Rom
    if len(ranks) > 1:
        x_smooth, y_smooth = catmull_rom_spline(x, y, samples=20)
    else:
        x_smooth, y_smooth = x, y

    dpi = 100
    fig_w = width_px / dpi
    fig_h = height_px / dpi

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)

    bg_color = "#0f172a"
    if background == "transparent":
        ax.set_facecolor("none")
        fig.patch.set_alpha(0.0)
    else:
        ax.set_facecolor(bg_color)
        fig.patch.set_facecolor(bg_color)

    # тёмный grid
    grid_color = "#00000055"
    ax.yaxis.grid(True, color=grid_color, linewidth=0.7)
    ax.xaxis.grid(True, color=grid_color, linewidth=0.7)

    # GLOW как в твоём generate_scoresaber_style_graph,
    # только немного усиленный, чтобы точно был виден
    glow_color = "#4CC9F0"
    glow_layers = [
        (22, 0.06),
        (18, 0.10),
        (14, 0.16),
        (10, 0.22),
    ]
    for lw, alpha in glow_layers:
        ax.plot(
            x_smooth,
            y_smooth,
            color=glow_color,
            linewidth=lw,
            alpha=alpha,
            solid_capstyle="round",
            solid_joinstyle="round",
        )

    # основная линия поверх
    ax.plot(
        x_smooth,
        y_smooth,
        color=glow_color,
        linewidth=2.5,
        solid_capstyle="round",
        solid_joinstyle="round",
    )

    # меньше ранг = выше на графике
    ax.invert_yaxis()

    # убираем рамки
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.tick_params(axis="x", colors="#A8B3CF", labelsize=8, length=0)
    ax.tick_params(axis="y", colors="#A8B3CF", labelsize=8, length=0)
    ax.set_xticklabels([])

    fig.subplots_adjust(left=0.03, right=0.99, top=0.97, bottom=0.08)

    buf = io.BytesIO()
    if background == "transparent":
        fig.savefig(buf, format="png", transparent=True)
    else:
        fig.savefig(buf, format="png", facecolor=fig.get_facecolor())
    plt.close(fig)

    buf.seek(0)
    img = Image.open(buf).convert("RGBA")
    return img

def compose_player_card_image(player_info: dict, output_path: str, width_px: int = 1600):
    """
    Генерирует только карточку игрока — ту же, что в Both,
    но сохраняет как отдельное PNG.
    """
    header_img = make_player_header_image(player_info, width_px)

    # Делает прозрачный фон вокруг, чтобы стиль совпадал с Both
    margin_x = 48
    margin_y = 48
    w, h = header_img.size

    total_w = w + margin_x * 2
    total_h = h + margin_y * 2

    canvas = Image.new("RGBA", (total_w, total_h), (0, 0, 0, 0))

    # Маска для скругления
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle([0, 0, w - 1, h - 1], radius=34, fill=255)

    bg = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    bg = Image.alpha_composite(bg, header_img)

    canvas.paste(bg, (margin_x, margin_y), mask)
    canvas.save(output_path, format="PNG")

    return output_path


def ensure_output_dir(path: str) -> str:
    if not path:
        return os.getcwd()
    os.makedirs(path, exist_ok=True)
    return path


def main():
    if len(sys.argv) < 2:
        print("No options JSON provided", file=sys.stderr)
        sys.exit(1)

    opts = json.loads(sys.argv[1])

    mode = opts.get("mode", "graph")  # both | graph | player
    player_id = str(opts.get("playerId", "")).strip()
    background = opts.get("background", "dark")
    fmt = (opts.get("format") or "png").lower()
    out_dir = ensure_output_dir(opts.get("outputDir", ""))

    if not player_id:
        raise SystemExit("playerId is required")

    base_name = f"{player_id}"
    if mode == "both":
        base_name += "_card_graph"
    elif mode == "player":
        base_name += "_player_card"
    else:
        base_name += "_graph"

    if mode in ("both", "player") or fmt not in ("png", "svg"):
        fmt = "png"

    out_path = os.path.join(out_dir, f"{base_name}.{fmt}")

    if mode == "graph":
        player_info = fetch_player_full(player_id)
        ranks = parse_histories(player_info)
        img = make_rank_graph_image(ranks, background=background)
        if fmt == "png":
            img.save(out_path, format="PNG")
        else:
            dpi = 100
            width_px, height_px = img.size
            fig_w = width_px / dpi
            fig_h = height_px / dpi
            fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
            ax.plot(range(len(ranks)), ranks, color="#60a5fa", linewidth=2.5)
            ax.invert_yaxis()
            ax.grid(True, color="#1f2933", linewidth=1, linestyle="-", alpha=0.9)
            for spine in ax.spines.values():
                spine.set_visible(False)
            fig.savefig(out_path, format="svg",
                        transparent=(background == "transparent"))
            plt.close(fig)
    elif mode == "both":
        player_info = fetch_player_full(player_id)
        ranks = parse_histories(player_info)
        compose_both_image(player_info, ranks, out_path)
    elif mode == "player":
        player_info = fetch_player_full(player_id)

        # Имя файла
        base_name = f"{player_id}_player_card"
        out_path = os.path.join(out_dir, f"{base_name}.png")

        compose_player_card_image(player_info, out_path)
    else:
        raise SystemExit(f"Unknown mode: {mode}")

    print(out_path)


if __name__ == "__main__":
    main()
