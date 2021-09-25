import time
from enum import Enum
import taichi as ti

ti.init(arch=ti.gpu)

grid_num = 100
state = ti.field(ti.u8, shape=(2, grid_num, grid_num))
state_idx = ti.field(ti.u32, shape=())

res = (1024, 1024)
zoom_rate = 1.2
zoom = 1.0
center_x = 0.0
center_y = 0.0
grid_size = 20.0
border_thickness = 0.05
time_step = 0.1
frame_buffer = ti.Vector.field(3, dtype=ti.u8, shape=res)


class RunningState(Enum):
    PAUSED = 1
    RUNNING = 2


@ti.kernel
def init():
    for i, j, k in state:
        state[i, j, k] = 0

    for i, j in frame_buffer:
        for k in ti.static(range(3)):
            frame_buffer[i, j][k] = 0x33

    center = ti.cast(grid_num / 2, ti.u32)
    state[0, center, center] = 1
    state[0, center - 1, center - 1] = 1
    state[0, center + 1, center] = 1
    state[0, center + 1, center - 1] = 1
    state[0, center, center + 1] = 1

    state_idx[None] = 0


@ti.func
def query_state(x, y):
    grid_x = x / grid_size + grid_num / 2
    grid_y = y / grid_size + grid_num / 2

    i = ti.cast(grid_x, ti.i32)
    j = ti.cast(grid_y, ti.i32)

    is_border = 0
    grid_state = 0
    if grid_x >= grid_num or grid_x < 0 or grid_y >= grid_num or grid_y < 0:
        pass
    else:
        grid_state = state[state_idx[None], ti.cast(i, ti.u32), ti.cast(j, ti.u32)]
        is_border = (grid_x - i) < border_thickness or (grid_x - i) > 1 - border_thickness or \
                    (grid_y - j) < border_thickness or (grid_y - j) > 1 - border_thickness

    return is_border, grid_state


@ti.kernel
def draw(pos_x: ti.f32, pos_y: ti.f32, zoom: ti.f32):
    for c, r in frame_buffer:
        x = pos_x + (c - res[0] / 2 + 0.5) / zoom
        y = pos_y + (r - res[1] / 2 + 0.5) / zoom

        is_border, grid_state = query_state(x, y)
        if is_border:
            for i in ti.static(range(3)):
                frame_buffer[c, r][i] = 0xcf
        elif grid_state:
            frame_buffer[c, r][0] = 0x7f
            frame_buffer[c, r][1] = 0xff
            frame_buffer[c, r][2] = 0x7f
        else:
            for i in ti.static(range(3)):
                frame_buffer[c, r][i] = 0x3f


@ti.kernel
def update():

    cur_state_idx = state_idx[None]
    nxt_state_idx = 1 - cur_state_idx

    for idx, i, j in state:
        if idx == cur_state_idx:
            # count alive neighbors
            alive = 0
            for dx in ti.static(range(-1, 2)):
                for dy in ti.static(range(-1, 2)):
                    if dx != 0 or dy != 0:
                        x = i + dx
                        y = j + dy
                        if 0 <= x < grid_num and 0 <= y < grid_num:
                            if state[cur_state_idx, x, y] == 1:
                                alive += 1

            # Cell rules
            if state[cur_state_idx, i, j] == 0:
                if alive == 3:
                    state[nxt_state_idx, i, j] = 1
                else:
                    state[nxt_state_idx, i, j] = 0
            else:
                if 2 <= alive <= 3:
                    state[nxt_state_idx, i, j] = 1
                else:
                    state[nxt_state_idx, i, j] = 0


if __name__ == "__main__":
    gui = ti.GUI("Game of Life", res)

    init()

    last_time = time.time()
    gui_state = RunningState.RUNNING

    while gui.running:
        if time.time() > last_time + time_step and gui_state == RunningState.RUNNING:
            last_time = time.time()
            update()
            state_idx[None] = state_idx[None] ^ 1

        for e in gui.get_events(gui.PRESS, gui.MOTION):
            if e.key == ti.GUI.LMB:
                # left click: record position
                mouse_x0, mouse_y0 = gui.get_cursor_pos()
                center_x0, center_y0 = center_x, center_y
            elif e.key == ti.GUI.WHEEL:
                # scroll: zoom
                mouse_x, mouse_y = gui.get_cursor_pos()
                if e.delta[1] > 0:
                    zoom_new = zoom * zoom_rate
                elif e.delta[1] < 0:
                    zoom_new = zoom / zoom_rate

                # disable zoom out because border aliasing
                if zoom_new < 0.7:
                    continue

                center_x += (mouse_x - 0.5) * res[0] * (1 / zoom - 1 / zoom_new)
                center_y += (mouse_y - 0.5) * res[1] * (1 / zoom - 1 / zoom_new)
                zoom = zoom_new
            elif e.key == ti.GUI.RMB:
                # right click: toggle the state of selected cell
                if gui_state == RunningState.RUNNING:
                    continue
                mouse_x, mouse_y = gui.get_cursor_pos()
                grid_x = (center_x + (mouse_x - 0.5) * res[0] / zoom) / grid_size + grid_num / 2
                grid_y = (center_y + (mouse_y - 0.5) * res[1] / zoom) / grid_size + grid_num / 2
                if grid_x >= grid_num or grid_x < 0 or grid_y >= grid_num or grid_y < 0:
                    pass
                else:
                    state[state_idx[None], int(grid_x), int(grid_y)] ^= 1
            elif e.key == ti.GUI.SPACE:
                if gui_state == RunningState.RUNNING:
                    gui_state = RunningState.PAUSED
                else:
                    gui_state = RunningState.RUNNING

        if gui.is_pressed(ti.GUI.LMB):
            # drag: move
            mouse_x, mouse_y = gui.get_cursor_pos()
            center_x = center_x0 + (mouse_x0 - mouse_x) * res[0] / zoom
            center_y = center_y0 + (mouse_y0 - mouse_y) * res[1] / zoom

        draw(center_x, center_y, zoom)
        gui.set_image(frame_buffer)
        gui.show()
