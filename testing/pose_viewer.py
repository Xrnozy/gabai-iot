import cv2, socket, struct, bisect, numpy as np

# ---------------- SETTINGS (change these) ----------------
PI_IP, PORT = "192.168.1.63", 5000   # the Pi's current IP
ROTATE_LEFT_180 = True               # True = turn the left camera 180 degrees
ROTATE_RIGHT_180 = False             # True = turn the right camera 180 degrees
CAM_LATENCY = 0.045                  # seconds from exposure to timestamp
BETA = 0.05                          # fusion gain (higher = trusts gravity more)
R_CAM_IMU = np.eye(3)                # IMU axes -> camera axes (set after mounting)
# ----------------------------------------------------------

RZ180 = np.diag([-1.0, -1.0, 1.0])

def recvall(s, n):
    b = bytearray()
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise ConnectionError
        b += c
    return bytes(b)

def madgwick(q, g, a, dt):
    q0, q1, q2, q3 = q
    gx, gy, gz = g
    step = np.zeros(4)
    n = np.linalg.norm(a)
    if n > 0:
        ax, ay, az = a / n
        f = np.array([2*(q1*q3-q0*q2)-ax, 2*(q0*q1+q2*q3)-ay, 2*(0.5-q1*q1-q2*q2)-az])
        J = np.array([[-2*q2, 2*q3, -2*q0, 2*q1],
                      [2*q1, 2*q0, 2*q3, 2*q2],
                      [0, -4*q1, -4*q2, 0]])
        step = J.T @ f
        step /= np.linalg.norm(step) + 1e-12
    qd = 0.5*np.array([-q1*gx-q2*gy-q3*gz, q0*gx+q2*gz-q3*gy,
                       q0*gy-q1*gz+q3*gx, q0*gz+q1*gy-q2*gx]) - BETA*step
    q = q + qd*dt
    return q / np.linalg.norm(q)

def slerp(a, b, u):
    d = np.dot(a, b)
    if d < 0:
        b, d = -b, -d
    if d > 0.9995:
        r = a + u*(b-a)
        return r/np.linalg.norm(r)
    th = np.arccos(d)
    return (np.sin((1-u)*th)*a + np.sin(u*th)*b) / np.sin(th)

def rot(q):
    w, x, y, z = q
    return np.array([[1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
                     [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
                     [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])

def pose_at(hist_t, hist_q, t):
    i = bisect.bisect_left(hist_t, t)
    if i == 0:
        return hist_q[0]
    if i >= len(hist_t):
        return hist_q[-1]
    u = (t-hist_t[i-1]) / (hist_t[i]-hist_t[i-1] + 1e-12)
    return slerp(hist_q[i-1], hist_q[i], u)

def draw(img, R_wc, label, Rci):
    ox, oy, L = 60, img.shape[0]-60, 45
    Rc = R_wc @ Rci.T                      # camera-to-world
    for v, col, name in ((0, (0, 0, 255), "X"), (1, (0, 255, 0), "Y"), (2, (255, 0, 0), "Z")):
        c = Rc.T[:, v]                     # world axis in camera frame
        cv2.line(img, (ox, oy), (int(ox+L*c[0]), int(oy+L*c[1])), col, 2)
        cv2.putText(img, name, (int(ox+L*c[0]), int(oy+L*c[1])), 0, 0.5, col, 1)
    roll = np.degrees(np.arctan2(Rc[2, 1], Rc[2, 2]))
    pitch = np.degrees(-np.arcsin(np.clip(Rc[2, 0], -1, 1)))
    yaw = np.degrees(np.arctan2(Rc[1, 0], Rc[0, 0]))
    cv2.putText(img, f"{label} r{roll:5.1f} p{pitch:5.1f} y{yaw:5.1f}",
                (10, 20), 0, 0.5, (0, 255, 255), 1)

s = socket.create_connection((PI_IP, PORT))
q = np.array([1.0, 0, 0, 0])
tprev = None
hist_t, hist_q = [], []
while True:
    n, tl, tr, ll, lr = struct.unpack("!IddII", recvall(s, 28))
    imu = np.frombuffer(recvall(s, n*56), dtype=np.float64).reshape(n, 7)
    jl, jr = recvall(s, ll), recvall(s, lr)
    for t, gx, gy, gz, ax, ay, az in imu:
        if tprev is not None and 0 < t-tprev < 0.1:
            q = madgwick(q, (gx, gy, gz), np.array([ax, ay, az]), t-tprev)
        tprev = t
        hist_t.append(t)
        hist_q.append(q.copy())
    del hist_t[:-2000], hist_q[:-2000]
    if not hist_t:
        continue
    imgs = []
    for ts, j, lab, flip in ((tl, jl, "L", ROTATE_LEFT_180), (tr, jr, "R", ROTATE_RIGHT_180)):
        im = cv2.imdecode(np.frombuffer(j, np.uint8), 1)
        if im is None:
            continue
        R = rot(pose_at(hist_t, hist_q, ts - CAM_LATENCY))
        Rci = RZ180 @ R_CAM_IMU if flip else R_CAM_IMU
        if flip:
            im = cv2.rotate(im, cv2.ROTATE_180)
        draw(im, R, lab, Rci)
        imgs.append(im)
    if len(imgs) == 2:
        cv2.imshow("stereo pose", np.hstack(imgs))
    if cv2.waitKey(1) == 27:
        break