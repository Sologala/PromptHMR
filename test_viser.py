from http import server
import time

from python_libs.chumpy.chumpy.ch_ops import floor
import viser
from prompt_hmr.vis.viser import viser_vis_world4d
from prompt_hmr.vis.traj import get_floor_fix_range

def main() -> None:
    server = viser.ViserServer()
    server.scene.world_axes.visible = True
    gui_up = server.gui.add_vector3(
        "Up Direction",
        initial_value=(0.0, 0.0, 1.0),
        step=0.01,
    )
    floor_arg = None

    gv, gf, _ = get_floor_fix_range((-100,  100), (-100, 100), scale=2, floor_color=None)
    server.scene.add_mesh_simple(
        f"/floor",
        vertices=gv,
        faces=gf,
        flat_shading=False,
        wireframe=True,
        color=(50, 50, 50),
    )

    @gui_up.on_update
    def _(_) -> None:
        server.scene.set_up_direction(gui_up.value)

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()