"""Shove the duck at t=3 s; report whether it is really upright again (gravity z < -0.95 and trunk
above 0.10 m, held 2 s) within `seconds` after the shove."""
import json, socket, sys, time
sock_path, push_file, seconds, speed = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
s = socket.socket(socket.AF_UNIX); s.connect(sock_path)
s.sendall((json.dumps({"jsonrpc": "2.0", "id": 1, "method": "robot.subscribe", "params": {"hz": 20}}) + "\n").encode())
t0 = time.monotonic(); pushed_at = None; down = False; upright_since = None; up_after = None; min_z = 9
for line in s.makefile():
    m = json.loads(line)
    if m.get("method") != "robot.state":
        continue
    t = time.monotonic() - t0
    p = m["params"]; g = p["safety"]["gravity"][2]; z = p["odom"]["position"][2]
    if pushed_at is None:
        if t > 3:
            if not (g < -0.95 and z > 0.10):
                print("NOT STANDING BEFORE THE PUSH"); break
            open(push_file, "w").write(speed); pushed_at = t
        continue
    min_z = min(min_z, z)
    if g > -0.5 or z < 0.07:
        down = True
    ok = g < -0.95 and z > 0.10
    if down and ok:
        upright_since = upright_since or t
        if t - upright_since >= 2.0:
            up_after = upright_since - pushed_at; break
    else:
        upright_since = None
    if t - pushed_at > seconds:
        break
print(f"fell={down} lowest_trunk={min_z:.3f} m up_after={'%.1f s' % up_after if up_after is not None else 'NO'}")
