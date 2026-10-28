"""The range half of an Ice Walk render (D22, 2026-10-28): the GPU, the snow cache, the per-range checks and the render of
clip frames R0..R1 from a built scene. ice_world.py builds the world and calls it (one-shot renders); ice_render.py opens a
world built once (ice_world.py --build-only, cached by the farm) and calls it, so a chunk no longer rebuilds the world.
The scene carries what the range needs as custom properties (ice_N, ice_FO, ice_speed, ice_sun_el, ice_snow, set by
ice_world.py before it saves)."""
import os, sys

import bpy


def use_gpu(sc):
    """Cycles on the box's GPU (OptiX, CUDA, or Metal on quintus); the device choice is a preference, not in the .blend."""
    prefs = bpy.context.preferences.addons["cycles"].preferences
    for t_ in ("OPTIX", "CUDA", "METAL"):          # METAL: the M5 (quintus) renders chunks too
        try:
            prefs.compute_device_type = t_; prefs.get_devices()
            if any(d.type == t_ for d in prefs.devices):
                break
        except TypeError:
            continue
    for d in prefs.devices:
        d.use = d.type != "CPU"
    sc.cycles.device = "GPU"


def point_snow(path):
    """The snow's external caches at PATH on this box (a .blend built on gruntus holds gruntus's path). Setting the path
    rescans the folder and resets the end frame to 999 (Blender 5.2; past it every flake froze, 2026-10-28), so each
    cache's range is put back after it and checked."""
    for ob in bpy.data.objects:
        for ps_ in ob.particle_systems:
            pc = ps_.point_cache
            if pc.use_external:
                r = (pc.frame_start, pc.frame_end)
                pc.filepath = os.path.abspath(path)
                pc.frame_start, pc.frame_end = r
                assert (pc.frame_start, pc.frame_end) == r, (ob.name, pc.frame_start, pc.frame_end, r)


def check_and_render(sc, out, mode, R0, R1, only=""):
    # measured from the built scene, so a launcher can stop before rendering long (user on the song, 2026-10-28: the snow
    # froze from 0:53 and the swifts flew out of frame, and nothing checked): the flakes' speed near the camera at the
    # range's first and last frames, and how many of the swifts' positions while drawn are in frame
    from bpy_extras.object_utils import world_to_camera_view
    cam, N, FO, speed = sc.camera, sc["ice_N"], sc["ice_FO"], list(sc["ice_speed"])

    def snow_speed(k):
        sc.frame_set(FO + k); dg_ = bpy.context.evaluated_depsgraph_get(); cp_ = cam.matrix_world.translation
        al = [q for q in bpy.data.objects["flurry"].evaluated_get(dg_).particle_systems[0].particles
              if q.alive_state == "ALIVE" and (q.location - cp_).length < 30]
        return sum(q.velocity.length for q in al) / max(1, len(al))

    snow_rep = "snow m/s %.2f %.2f" % (snow_speed(R0), snow_speed(R1)) if sc["ice_snow"] else "snow off"
    seen_, all_ = 0, 0
    birds = [o for o in bpy.data.objects if o.name.startswith("swift_bird")]
    for k in range(R0, R1 + 1, 2):
        if birds:
            sc.frame_set(FO + k)
        for b_ in birds:
            if not b_.hide_render:
                v_ = world_to_camera_view(sc, cam, b_.matrix_world.translation)
                all_ += 1; seen_ += (0 <= v_.x <= 1 and 0 <= v_.y <= 1 and v_.z > 0)
    swift_rep = "swifts in frame %d/%d" % (seen_, all_)
    print("ice check; range", R0, R1, ";", snow_rep, ";", swift_rep, flush=True)
    if os.environ.get("CHECK_ONLY"):              # the checks alone, no render (a preflight)
        sys.exit(0)
    sc.render.filepath = f"{out}/{mode}/"
    if only:
        for f in (int(v) for v in only.split(",")):
            sc.frame_set(f + FO); sc.render.filepath = f"{out}/{mode}/{f:04d}.png"
            bpy.ops.render.render(write_still=True)
    else:
        sc.frame_start, sc.frame_end = FO + R0, FO + R1      # (the end too: it was left at the song's end after the bake)
        bpy.ops.render.render(animation=True)
        for k in range(R0, R1 + 1):                # scene frame FO+k -> file k
            os.replace(f"{out}/{mode}/{FO + k:04d}.png", f"{out}/{mode}/{k:04d}.png")
    print("ice ok", out, N, "frames; range", R0, R1, "; layers:",          # so a launcher can check what it built
          " ".join(f"{k}={sum(o.name.startswith(k) for o in bpy.data.objects)}" for k in ("halo", "clouds", "drift", "chime", "crack", "swift_bird")),
          "; sun_el", sc["ice_sun_el"], "look", sc.view_settings.look,
          "; speed m/s %.1f-%.1f" % (min(speed[R0 - 1:R1]), max(speed[R0 - 1:R1])), ";", snow_rep, ";", swift_rep)
